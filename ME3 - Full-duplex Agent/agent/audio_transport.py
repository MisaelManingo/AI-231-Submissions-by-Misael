#!/usr/bin/env python3
"""
Hardware Audio Transport for G-MARK Micro Go USB
Features:
- Separate Input & Output FrameProcessors so pipeline linking never conflicts
- Fixed 48,000 Hz hardware capture & playback (required by ALSA hardware driver)
- Fast 3:1 decimation to 16,000 Hz for Silero VAD & Gemini Live
- Real-time upsampling and mono-to-stereo expansion for 48,000 Hz 2-channel speaker output
- Live continuous LCD VU meter feedback from incoming microphone audio
- Instant buffer flush on barge-in interruption (user voice stops agent audio immediately)
- Speaker state tracking for software acoustic echo suppression
"""

import time
import queue
import asyncio
import threading
import numpy as np
import scipy.signal
import sounddevice as sd

from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    OutputAudioRawFrame,
    TTSAudioRawFrame,
    TTSStartedFrame,
    TTSStoppedFrame,
    LLMFullResponseEndFrame,
    InterruptionFrame,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from config import AUDIO_DEVICE_NAME, AUDIO_HARDWARE_RATE, AUDIO_MODEL_RATE
from config import AEC_ENABLED, AEC_PARTITIONS, AEC_MU, AEC_CONVERGED_ERLE_DB
from config import MUSIC_TO_HDMI, MUSIC_HDMI_MAX_GAIN, MUSIC_MAX_GAIN
from hardware.music_hdmi import HdmiMusicOutput
from .aec import PlaybackReference, EchoCanceller
from hardware.music_player import MusicPlayer


def find_audio_device_indices():
    """Finds input and output device indices matching G-MARK Micro Go USB."""
    in_idx = None
    out_idx = None
    devices = sd.query_devices()
    for idx, d in enumerate(devices):
        name = d.get("name", "")
        if AUDIO_DEVICE_NAME.lower() in name.lower():
            if d.get("max_input_channels", 0) > 0 and in_idx is None:
                in_idx = idx
            if d.get("max_output_channels", 0) > 0 and out_idx is None:
                out_idx = idx

    return in_idx, out_idx


class GMarkAudioInputTransport(FrameProcessor):
    """
    Input processor: reads audio from microphone stream callback and pushes
    InputAudioRawFrame downstream into VAD and Gemini Live.
    """
    def __init__(self, hardware_manager=None):
        super().__init__()
        self.hw_mgr = hardware_manager
        self.barge_in_active = False
        self._latest_rms = 0.0
        self._vu_thread = None

    def on_audio_data(self, pcm_16k: bytes, rms: float):
        """Called by audio callback in main thread to update VU meter."""
        # FIX (mic cropping): the LCD flush does slow I2C writes (~100-250 ms). Doing it inside the
        # PortAudio callback made the input overrun and drop mic chunks (arrival gaps ~246 ms).
        # The callback now only stores the latest RMS; a daemon thread pushes it to the LCD.
        # To revert: restore the direct `self.hw_mgr.update_vu(rms)` call here and delete _vu_worker.
        self._latest_rms = rms
        if self.hw_mgr and self._vu_thread is None:
            self._vu_thread = threading.Thread(target=self._vu_worker, daemon=True)
            self._vu_thread.start()

    def _vu_worker(self):
        while True:
            try:
                self.hw_mgr.update_vu(self._latest_rms)
            except Exception:
                pass
            time.sleep(0.15)

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class GMarkAudioOutputTransport(FrameProcessor):
    """
    Output processor: receives TTSAudioRawFrame from Gemini Live and renders
    audio through the G-MARK USB speaker. Handles barge-in cancellation.
    """
    def __init__(self, hardware_manager=None, benchmark_logger=None):
        super().__init__()
        self.hw_mgr = hardware_manager
        self.benchmark = benchmark_logger

        self.playback_queue = queue.Queue(maxsize=150)
        self.is_playing = False
        self.model_speaking = False
        self.playback_start_ts = 0.0
        self.last_audio_played_ts = 0.0
        self.lock = threading.Lock()
        self.hw_rate = AUDIO_HARDWARE_RATE
        self.publisher = None
        # LISTEN-PROBE: 1.0 = play, 0.0 = paused (queued speech is kept, 20 ms fades). See config.DUCK_PROBE_ENABLED.
        self.pause_target = 1.0
        self.pause_gain = 1.0

    def set_listen_pause(self, on: bool):
        self.pause_target = 0.0 if on else 1.0

    def peek_upcoming_rms(self, max_chunks: int = 2):
        """RMS of the next queued speaker audio (what plays right after a pause starts); None if nothing queued."""
        try:
            items = list(self.playback_queue.queue)[:max_chunks]
            if not items:
                return None
            x = np.concatenate([c[:, 0] for c in items])
            return float(np.sqrt(np.mean(x ** 2)))
        except Exception:
            return None

    def cancel_playback_immediately(self):
        """Barge-in: clears queued speaker frames and mutes output instantly."""
        t_start = time.perf_counter()
        was_playing = False
        with self.lock:
            queue_had_items = not self.playback_queue.empty()
            while not self.playback_queue.empty():
                try:
                    self.playback_queue.get_nowait()
                except queue.Empty:
                    break
            was_playing = self.is_playing or self.model_speaking or queue_had_items
            self.is_playing = False
            self.model_speaking = False
            self.playback_start_ts = 0.0
            self.last_audio_played_ts = 0.0
            self.pause_target = 1.0   # a cancelled playback must not leave the next answer paused
            self.pause_gain = 1.0

        t_silent = time.perf_counter()
        if was_playing:
            if self.benchmark:
                self.benchmark.record_barge_in(t_start, t_silent)

            if self.hw_mgr:
                try:
                    self.hw_mgr.on_barge_in()
                except Exception:
                    pass

            if getattr(self, "publisher", None):
                self.publisher.set_state("idle")

            print(f"⚡ [BARGE-IN]: Playback cancelled in {(t_silent - t_start)*1000.0:.2f}ms.")

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            self.cancel_playback_immediately()
            await self.push_frame(frame, direction)

        elif isinstance(frame, TTSStartedFrame):
            with self.lock:
                self.model_speaking = True
            await self.push_frame(frame, direction)

        elif isinstance(frame, (TTSStoppedFrame, LLMFullResponseEndFrame)):
            with self.lock:
                self.model_speaking = False
            if getattr(self, "publisher", None):
                self.publisher.set_state("idle")
            await self.push_frame(frame, direction)

        elif isinstance(frame, (TTSAudioRawFrame, OutputAudioRawFrame)):
            audio_bytes = frame.audio
            sr = frame.sample_rate

            with self.lock:
                self.model_speaking = True

            if self.benchmark:
                self.benchmark.record_model_first_audio()

            if self.hw_mgr:
                self.hw_mgr.on_speaking()

            # Decode int16 PCM to float32 with speaker volume gain to avoid microphone saturation
            SPEAKER_GAIN = 0.45
            audio_arr = (np.frombuffer(audio_bytes, dtype=np.int16).astype(np.float32) / 32768.0) * SPEAKER_GAIN

            # Publish speaker level outside PortAudio callback (strict callback isolation)
            if getattr(self, "publisher", None) and len(audio_arr) > 0:
                out_rms = float(np.sqrt(np.mean(audio_arr ** 2)))
                self.publisher.update_speaker_level(out_rms)

            # Upsample to 48,000 Hz
            if sr == 24000:
                audio_48k = np.repeat(audio_arr, 2)
            elif sr == 16000:
                audio_48k = np.repeat(audio_arr, 3)
            else:
                target_len = int(len(audio_arr) * self.hw_rate / sr)
                audio_48k = np.interp(
                    np.linspace(0, len(audio_arr) - 1, target_len),
                    np.arange(len(audio_arr)),
                    audio_arr
                ).astype(np.float32)

            # Mono to stereo (2 channels for G-MARK speaker)
            stereo_48k = np.column_stack((audio_48k, audio_48k))

            if self.benchmark and not self.is_playing:
                self.benchmark.record_speaker_first_audio()

            try:
                self.playback_queue.put_nowait(stereo_48k)
            except queue.Full:
                pass

            await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)


class GMarkAudioSystem:
    """
    Coordinates hardware sounddevice streams and provides separate .input and .output
    processors for the Pipecat pipeline.
    """
    def __init__(self, hardware_manager=None, benchmark_logger=None):
        self.hw_mgr = hardware_manager
        self.benchmark = benchmark_logger

        self.in_idx, self.out_idx = find_audio_device_indices()
        print(f"[AUDIO SYSTEM]: Selected Input={self.in_idx}, Output={self.out_idx}")

        self.hw_rate = AUDIO_HARDWARE_RATE  # 48000 Hz
        self.model_rate = AUDIO_MODEL_RATE  # 16000 Hz

        # Create separate input and output processors
        self.input = GMarkAudioInputTransport(hardware_manager=hardware_manager)
        self.output = GMarkAudioOutputTransport(hardware_manager=hardware_manager, benchmark_logger=benchmark_logger)

        # Dedicated background music player
        self.music_player = MusicPlayer()
        self.music_hdmi = None  # HdmiMusicOutput when music is routed to the HDMI speaker (see hardware/music_hdmi.py)

        # ECHO CANCELLATION: output callback records what the speaker plays (incl. music mix);
        # the VAD processor uses it to subtract the echo from the mic. None => legacy behaviour.
        self.playback_ref = PlaybackReference()
        self.aec = (EchoCanceller(self.playback_ref, partitions=AEC_PARTITIONS, mu=AEC_MU,
                                  converged_erle_db=AEC_CONVERGED_ERLE_DB) if AEC_ENABLED else None)

        self.input_stream = None
        self.output_stream = None
        self.loop = None
        self.running = True

    @property
    def is_playing(self) -> bool:
        with self.output.lock:
            now = time.perf_counter()
            # If the output queue still has audio waiting to play
            if not self.output.playback_queue.empty():
                return True
            # If audio was played within the last 120ms (decay window)
            if now - self.output.last_audio_played_ts < 0.12:
                return True
            # If output callback is currently writing frames
            if self.output.is_playing:
                return True
            # Auto-healing watchdog: if model_speaking is True but queue is empty and silent for > 0.25s,
            # it is a stale flag (e.g. dropped TTSStoppedFrame after server interruption). Clear it!
            if self.output.model_speaking:
                if now - self.output.last_audio_played_ts >= 0.25:
                    self.output.model_speaking = False
                    return False
                return True
            return False

    @property
    def model_speaking(self) -> bool:
        with self.output.lock:
            return bool(self.output.model_speaking)

    @property
    def playback_duration(self) -> float:
        with self.output.lock:
            if self.output.is_playing and self.output.playback_start_ts > 0.0:
                return time.perf_counter() - self.output.playback_start_ts
        return 0.0

    def cancel_playback_immediately(self):
        self.output.cancel_playback_immediately()

    async def start(self, loop):
        """Opens 48 kHz hardware streams and begins pushing audio frames."""
        self.loop = loop
        print("[AUDIO SYSTEM]: Opening 48,000 Hz sounddevice streams on G-MARK Micro Go...")

        # Turn off noisy hardware Auto Gain Control on ALSA Card 2
        try:
            import subprocess
            subprocess.run(["amixer", "-c", "2", "set", "Auto Gain Control", "off"], capture_output=True, timeout=1)
        except Exception:
            pass

        # 1. 7.5 kHz 4th-order Butterworth anti-aliasing low-pass filter at 48 kHz with continuous state
        sos_lpf = scipy.signal.butter(4, 7500, btype="lowpass", fs=self.hw_rate, output="sos")
        lpf_zi = scipy.signal.sosfilt_zi(sos_lpf)

        # 2. 100 Hz 2nd-order Butterworth high-pass filter at 16 kHz: eliminates mechanical fan rumble
        sos_hpf = scipy.signal.butter(2, 100, btype="highpass", fs=self.model_rate, output="sos")
        hpf_zi = scipy.signal.sosfilt_zi(sos_hpf)

        mic_block_idx = 0

        def input_callback(indata, frames, time_info, status):
            nonlocal lpf_zi, hpf_zi, mic_block_idx
            if status:
                pass
            if not self.running:
                return

            # Anti-aliased downsampling: 48 kHz -> 16 kHz using stateful continuous IIR filtering
            # Eliminates edge clicks and block boundary discontinuities
            samples_48k = indata[:, 0]
            filt_48k, lpf_zi = scipy.signal.sosfilt(sos_lpf, samples_48k, zi=lpf_zi)
            samples_16k = filt_48k[::3].astype(np.float32)

            # High-pass filter (100 Hz): strips sub-audible desk vibrations and electrical rumble
            filtered_16k, hpf_zi = scipy.signal.sosfilt(sos_hpf, samples_16k, zi=hpf_zi)

            # Calibrated digital gain (1.15x) based on real voice measurements to avoid saturation clipping
            boosted = np.clip(filtered_16k * 1.15, -0.95, 0.95)

            # Compute RMS for VU meter (reflects real room acoustics)
            rms = float(np.sqrt(np.mean(boosted ** 2)))
            self.input.on_audio_data(boosted, rms)

            mic_block_idx += 1

            # Convert to 16-bit PCM for downstream VAD and Gemini Live
            pcm_16k = (boosted * 32767).astype(np.int16).tobytes()

            if self.loop and self.loop.is_running():
                frame = InputAudioRawFrame(
                    audio=pcm_16k,
                    sample_rate=self.model_rate,
                    num_channels=1
                )
                frame.mic_idx = mic_block_idx  # AEC alignment stamp (see agent/aec.py)
                asyncio.run_coroutine_threadsafe(self.input.push_frame(frame), self.loop)

        def output_callback(outdata, frames, time_info, status):
            if status:
                pass

            outdata.fill(0)
            filled = 0
            # LISTEN-PROBE pause: while paused (and fully faded out) do not consume the queue at all.
            pg0, pg1 = self.output.pause_gain, self.output.pause_target
            consume = not (pg0 <= 0.0 and pg1 <= 0.0)
            while consume and filled < frames:
                try:
                    chunk = self.output.playback_queue.get_nowait()
                    chunk_len = len(chunk)
                    needed = frames - filled
                    if chunk_len <= needed:
                        outdata[filled:filled + chunk_len] = chunk
                        filled += chunk_len
                    else:
                        outdata[filled:frames] = chunk[:needed]
                        self.output.playback_queue.queue.appendleft(chunk[needed:])
                        filled = frames
                except queue.Empty:
                    break

            if pg0 != 1.0 or pg1 != 1.0:
                if filled > 0:
                    outdata[:filled] *= np.linspace(pg0, pg1, filled, dtype=np.float32)[:, None]
                self.output.pause_gain = pg1

            now = time.perf_counter()
            with self.output.lock:
                if filled > 0:
                    self.output.last_audio_played_ts = now
                    if not self.output.is_playing:
                        self.output.is_playing = True
                        self.output.playback_start_ts = now
                else:
                    queue_has_data = not self.output.playback_queue.empty()
                    decay_active = (now - self.output.last_audio_played_ts < 0.12)
                    if not self.output.model_speaking and not queue_has_data and not decay_active:
                        if self.output.is_playing:
                            self.output.is_playing = False
                            self.output.playback_start_ts = 0.0
                            if self.hw_mgr:
                                try:
                                    self.hw_mgr.on_idle()
                                except Exception:
                                    pass

            # 2. Mix Background Music (Ducked to 15% when Gemini is speaking)
            # (skipped while the HDMI output is alive: music is then played on the HDMI speaker instead)
            if (hasattr(self, "music_player") and self.music_player and self.music_player.is_playing
                    and not (self.music_hdmi and self.music_hdmi.alive)):
                is_ducked = (filled > 0) or getattr(self.output, "model_speaking", False)
                music_chunk = self.music_player.get_audio_chunk(frames, ducked=is_ducked)
                if music_chunk is not None:
                    outdata += music_chunk
                    np.clip(outdata, -1.0, 1.0, out=outdata)

            # AEC reference: exactly what goes to the speaker (post-gain, post-music), mono 16 kHz.
            # Cheap by design (no FFT/locks): this runs in the PortAudio callback.
            n3 = (frames // 3) * 3
            self.playback_ref.write(outdata[:n3, 0].reshape(-1, 3).mean(axis=1))

        # Open ALSA hardware streams
        self.input_stream = sd.InputStream(
            device=self.in_idx,
            channels=1,
            samplerate=self.hw_rate,
            blocksize=1536,  # 32ms @ 48 kHz (gives 512 samples @ 16 kHz)
            dtype="float32",
            callback=input_callback
        )
        self.output_stream = sd.OutputStream(
            device=self.out_idx,
            channels=2,
            samplerate=self.hw_rate,
            blocksize=960,  # 20ms @ 48 kHz
            dtype="float32",
            callback=output_callback
        )

        self.input_stream.start()
        self.output_stream.start()
        if MUSIC_TO_HDMI:  # separate thread + aplay process; any failure leaves music on the USB speaker
            try:
                self.music_hdmi = HdmiMusicOutput(
                    self.music_player,
                    lambda: self.output.is_playing or getattr(self.output, "model_speaking", False),
                    gain_scale=MUSIC_HDMI_MAX_GAIN / MUSIC_MAX_GAIN)
                self.music_hdmi.start()
            except Exception as e:
                print(f"🎵 [MUSIC HDMI]: disabled ({e})", flush=True)
                self.music_hdmi = None
        print("[AUDIO SYSTEM]: Microphone capture and speaker playback streams active.")

    def close(self):
        self.running = False
        if self.music_hdmi:
            self.music_hdmi.stop()
        if self.input_stream:
            try:
                self.input_stream.stop()
                self.input_stream.close()
            except Exception:
                pass
        if self.output_stream:
            try:
                self.output_stream.stop()
                self.output_stream.close()
            except Exception:
                pass
