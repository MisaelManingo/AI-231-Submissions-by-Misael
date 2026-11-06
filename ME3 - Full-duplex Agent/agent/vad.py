#!/usr/bin/env python3
"""
On-Device Silero VAD & Barge-In Processor
Runs Silero VAD (ONNX) on the Raspberry Pi 5 to:
1. Detect user speech start & end boundaries in real time
2. Differentiate user voice from speaker echo (Echo Gate)
3. Trigger immediate barge-in interruption when user speaks during playback
4. Record high-precision end-of-speech timestamps for latency benchmarking
"""

import os
import time
import asyncio
import collections
import numpy as np
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState
from pipecat.frames.frames import (
    Frame,
    InputAudioRawFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    InterruptionFrame
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from config import (
    VAD_CONFIDENCE_THRESHOLD, VAD_STOP_SPEECH_MS,
    BARGE_IN_CONFIDENCE_THRESHOLD, BARGE_IN_RMS_THRESHOLD,
    AEC_BARGE_CONFIDENCE, AEC_BARGE_MIN_RMS, AEC_BARGE_FLOOR_MULT, BARGE_IN_MIN_MS,
    DUCK_PROBE_ENABLED, PROBE_INTERVAL_MS, PROBE_PAUSE_MS, PROBE_HOLD_MAX_MS,
    ECHO_DELAY_MS, ECHO_COUPLING, PROBE_ECHO_OK, PROBE_MIN_RMS, PROBE_QUIET_RMS, PROBE_DEFER_MS
)

# DIAGNOSTIC ONLY (no behavior change): set ME3_AUDIO_TRACE=1 to print per-turn frame counts/durations.
# Remove this flag and the `if TRACE:` blocks once the audio-cropping bug is found.
TRACE = os.getenv("ME3_AUDIO_TRACE", "0") == "1"


class LocalVADBargeInProcessor(FrameProcessor):
    """
    Real-time local VAD and barge-in controller.
    """
    def __init__(self, audio_transport, hardware_manager=None, benchmark_logger=None):
        super().__init__()
        self.transport = audio_transport
        self.hw_mgr = hardware_manager
        self.benchmark = benchmark_logger

        # Initialize Silero VAD Analyzer with conversational pause tolerance (750ms)
        stop_secs = (VAD_STOP_SPEECH_MS / 1000.0) if VAD_STOP_SPEECH_MS else 0.75
        self.vad = SileroVADAnalyzer(
            params=VADParams(
                confidence=VAD_CONFIDENCE_THRESHOLD,
                start_secs=0.10,
                stop_secs=stop_secs
            )
        )
        self.vad.set_sample_rate(16000)

        self.user_speaking = False
        self.speech_start_ts = 0.0
        self.last_speech_ts = 0.0
        self.onset_streak = 0
        self.barge_in_streak = 0

        # Rolling pre-roll buffer: 10 chunks of 32ms (~320ms)
        # Matches tests/calibrate_vad_splicing.py exactly: preserves the entire consonant onset
        # of words like 'Turn...', 'Make...', 'Play...' so Gemini receives full acoustic fidelity.
        self.preroll_buffer = collections.deque(maxlen=10)
        self._watchdog_task = None
        # TRACE counters (diagnostic only)
        self._tr_last_arrival = 0.0
        self._tr_in = 0
        self._tr_out = 0
        self._tr_pre = 0
        self._tr_onset_wall = 0.0
        self._tr_max_gap = 0.0

        # Echo suppression & barge-in thresholds
        # Note: G-MARK mic & speaker share the same USB housing.
        # Hardware GPIO 5 button provides instant zero-echo barge-in.
        self.echo_vad_threshold = BARGE_IN_CONFIDENCE_THRESHOLD
        self.barge_in_rms_threshold = BARGE_IN_RMS_THRESHOLD  # Intentional loud speech / close proximity
        self.publisher = None

        # BACKCHANNEL TOLERANCE: voice barge-in needs BARGE_IN_MIN_MS of sustained speech, so short
        # sounds over the agent ('yeah', 'uh-huh', cough) are ignored. Revert: ME3_BACKCHANNEL=0.
        self.barge_in_min_frames = max(2, -(-BARGE_IN_MIN_MS // 32))
        # While the streak builds, keep the (echo-cancelled) frames so a real barge-in does not
        # lose its first words ('wait, ...'). Flushed to Gemini when the barge-in is confirmed.
        self._bc_buffer = collections.deque(maxlen=self.barge_in_min_frames + 4)
        # sliding window of 'this frame looked like user speech'; barge-in when ~65% of the window is voiced
        self._bc_win = collections.deque(maxlen=self.barge_in_min_frames)
        self._bc_need = -(-self.barge_in_min_frames * 65 // 100) if self.barge_in_min_frames > 2 else 2
        self._bc_peak = 0
        self._pb_frames = 0       # frames since the speaker became active (residual calibration)
        self._res_floor = 0.01    # learned level of leftover echo after AEC
        self._res_ratio = 0.3     # leftover echo relative to the predicted echo (grows with agent loudness)
        self._tr_aec_ts = 0.0

        # LISTEN-PROBE state (see config.DUCK_PROBE_ENABLED): none -> probe (agent paused briefly) -> hold
        self._pr_state = "none"
        self._pr_n = 0            # speaker-active frames seen
        self._pr_t0 = 0           # frame number when the current pause started
        self._pr_since = 0        # frames since the last echo-free (listenable) frame
        self._pr_quiet = 0        # consecutive listenable frames without user speech while holding
        self._pr_peak = 0         # most voiced frames seen in the current episode (for the backchannel log)
        self._pr_total = 0        # voiced echo-free frames counted while holding (gaps between phrases allowed)

    # ------------------------------------------------------------------
    # LISTEN-PROBE helpers
    # ------------------------------------------------------------------
    def _probe_ready(self, mic_idx):
        return (DUCK_PROBE_ENABLED and mic_idx is not None and self.transport is not None
                and getattr(self.transport, "playback_ref", None) is not None
                and hasattr(getattr(self.transport, "output", None), "set_listen_pause"))

    def _predicted_echo_rms(self, mic_idx):
        """Echo level the mic should currently hear, from what the speaker played ~ECHO_DELAY ago."""
        d = int(ECHO_DELAY_MS * 16)
        seg = self.transport.playback_ref.get(mic_idx * 512 - d - 512, 1024)  # aligned block + the one before it
        env = max(float(np.sqrt(np.mean(seg[:512] ** 2))), float(np.sqrt(np.mean(seg[512:] ** 2))))
        return ECHO_COUPLING * env

    def _probe_release(self, why=""):
        if self._pr_state != "none":
            self.transport.output.set_listen_pause(False)
            if TRACE:
                print(f"[TRACE PROBE] release ({why}) after {(self._pr_n - self._pr_t0)*32}ms", flush=True)
        self._pr_state = "none"
        self._pr_quiet = 0
        self._pr_peak = 0

    def _probe_defer(self, out):
        """True while a probe is due but the next queued audio is loud speech: wait for a quiet moment (max PROBE_DEFER_MS)."""
        if self._pr_since * 32 >= PROBE_INTERVAL_MS + PROBE_DEFER_MS:
            return False
        up = out.peek_upcoming_rms() if hasattr(out, "peek_upcoming_rms") else None
        return up is not None and up > PROBE_QUIET_RMS

    def _probe_step(self, evaluable):
        """Pause/resume the agent so that echo-free listening windows exist. Runs once per mic frame."""
        out = self.transport.output
        self._pr_n += 1
        self._pr_since = 0 if evaluable else self._pr_since + 1
        voiced = sum(self._bc_win)
        self._pr_peak = max(self._pr_peak, voiced)
        held = (self._pr_n - self._pr_t0) * 32

        if self._pr_state == "none":
            if evaluable and voiced >= 2:
                # loud speech inside a natural pause of the agent: stop talking and keep listening
                out.set_listen_pause(True)
                self._pr_state, self._pr_t0, self._pr_quiet, self._pr_total = "hold", self._pr_n, 0, voiced
            elif self._pr_since * 32 >= PROBE_INTERVAL_MS and not self._probe_defer(out):
                out.set_listen_pause(True)    # no echo-free window for a while: make one (at a word boundary)
                self._pr_state, self._pr_t0 = "probe", self._pr_n
                self._bc_win.clear()
                self._pr_peak = 0
                if TRACE:
                    print("[TRACE PROBE] pause (listening window)", flush=True)
        elif self._pr_state == "probe":
            if voiced >= 2:
                self._pr_state, self._pr_quiet, self._pr_total = "hold", 0, voiced   # someone is talking: keep the agent paused
            elif held >= PROBE_PAUSE_MS:
                self._probe_release("nothing heard")
                self._pr_since = 0
        elif self._pr_state == "hold":
            if evaluable:
                if self._bc_win and self._bc_win[-1]:
                    self._pr_quiet = 0
                    self._pr_total += 1
                else:
                    self._pr_quiet += 1
            if self._pr_quiet >= 14 or held >= PROBE_HOLD_MAX_MS:
                if self._pr_peak >= 3:
                    print(f"💬 [BACKCHANNEL]: ignored short sound over agent (~{self._pr_peak*32}ms voiced < {self._bc_need*32}ms needed)")
                self._probe_release("short sound" if self._pr_quiet >= 14 else "max hold")
                self._pr_since = 0
                self._bc_win.clear()

    async def process_frame(self, frame: Frame, direction: FrameDirection):
        await super().process_frame(frame, direction)

        if isinstance(frame, InterruptionFrame):
            # Barge-in / Interruption from server or hardware button
            if self.transport:
                self.transport.cancel_playback_immediately()
            self.user_speaking = False
            self.barge_in_streak = 0
            self.onset_streak = 0
            self.preroll_buffer.clear()
            self._bc_buffer.clear()
            self._bc_win.clear()
            self._pr_state = "none"  # transport.cancel_playback_immediately() already un-paused the output
            if self._watchdog_task and not self._watchdog_task.done():
                self._watchdog_task.cancel()
            if self.hw_mgr:
                try:
                    self.hw_mgr.on_idle()
                except Exception:
                    pass
            await self.push_frame(frame, direction)
            return

        if isinstance(frame, InputAudioRawFrame):
            pcm = frame.audio
            samples = np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(samples ** 2)))

            if TRACE:
                _t = time.perf_counter()
                if self._tr_last_arrival:
                    _gap = _t - self._tr_last_arrival
                    self._tr_max_gap = max(self._tr_max_gap, _gap)
                    if _gap > 0.100:  # expected cadence is 32 ms
                        print(f"[TRACE VAD] frame arrival gap {_gap*1000:.0f}ms (speaking={self.user_speaking})", flush=True)
                self._tr_last_arrival = _t

            speaker_active = self.transport.is_playing if self.transport else False

            # -----------------------------------------------------------------
            # 1. SPEAKER ACTIVE (Agent is outputting voice through speaker)
            # -----------------------------------------------------------------
            if speaker_active:
                if TRACE and self.user_speaking:
                    print(f"[TRACE VAD] ECHO-GATE ABORT mid-utterance: pushed {self._tr_out} frames "
                          f"({self._tr_out*32}ms) downstream, NO UserStoppedSpeakingFrame will be sent", flush=True)

                self._pb_frames += 1
                send_frame = frame  # what Gemini gets if this turns into a barge-in
                aec = getattr(self.transport, "aec", None) if self.transport else None
                mic_idx = getattr(frame, "mic_idx", None)
                residual_ok = False
                probe_mode = self._probe_ready(mic_idx)
                evaluable = False
                if (not probe_mode) and aec is not None and mic_idx is not None:
                    # ECHO CANCELLATION: subtract the predicted speaker echo; decide on what is left.
                    res = aec.process(mic_idx, samples)
                    res_pcm = (np.clip(res, -1.0, 1.0) * 32767).astype(np.int16).tobytes()
                    residual_ok = aec.converged  # unconverged filter => keep the legacy raw-mic rule
                    if residual_ok:
                        send_frame = InputAudioRawFrame(
                            audio=res_pcm, sample_rate=frame.sample_rate, num_channels=frame.num_channels
                        )
                    if TRACE and time.perf_counter() - self._tr_aec_ts > 1.0:
                        self._tr_aec_ts = time.perf_counter()
                        print(f"[TRACE AEC] erle={aec.erle_db:.1f}dB converged={aec.converged} "
                              f"echo_delay~{aec.peak_delay_ms():.0f}ms res_floor={self._res_floor:.3f}", flush=True)

                if probe_mode:
                    # LISTEN-PROBE: only frames where the predicted speaker echo is small are trusted.
                    echo_est = self._predicted_echo_rms(mic_idx)
                    evaluable = echo_est <= PROBE_ECHO_OK
                    conf = float(self.vad.voice_confidence(pcm)[0]) if (evaluable and len(pcm) >= 512) else 0.0
                    # hysteresis: once the agent is paused for a listener, speech only needs conf >= 0.40 to count
                    conf_thr = 0.40 if self._pr_state == "hold" else 0.60
                    is_barge = evaluable and conf >= conf_thr and rms >= max(PROBE_MIN_RMS, 2.0 * echo_est + 0.04)
                    if TRACE and self._pr_state != "none":
                        print(f"[TRACE PROBE] state={self._pr_state} echo_est={echo_est:.3f} eval={evaluable} conf={conf:.2f} rms={rms:.3f} voiced={is_barge}", flush=True)
                elif residual_ok:
                    res_rms = float(np.sqrt(np.mean(res ** 2)))
                    y_rms = aec.last_echo_rms
                    conf = float(self.vad.voice_confidence(res_pcm)[0]) if len(res_pcm) >= 512 else 0.0
                    # Voice must stand clearly above the learned leftover-echo level.
                    is_barge = (
                        self._pb_frames > 10  # first ~320ms of playback only calibrates the floor
                        and conf >= AEC_BARGE_CONFIDENCE
                        and res_rms >= max(AEC_BARGE_MIN_RMS, max(AEC_BARGE_FLOOR_MULT * self._res_floor, 1.5 * self._res_ratio * y_rms))
                    )
                    # Floor = level of leftover echo. Learn it fast during the first ~320ms of playback,
                    # then only from audio that does NOT look like speech (otherwise the user's own quiet
                    # words would raise the floor and desensitise barge-in). Speech-like frames: tiny rate.
                    # Loud passages leave proportionally more echo, so also learn residual/echo-estimate.
                    rate = 0.2 if self._pb_frames <= 10 else (0.03 if conf < 0.5 else 0.002)
                    cap = 3 * self._res_floor + 0.01 if self._pb_frames > 10 else 1.0
                    self._res_floor += rate * (min(res_rms, cap) - self._res_floor)
                    if y_rms > 0.005:
                        self._res_ratio += rate * (min(res_rms / y_rms, 3 * self._res_ratio + 0.1) - self._res_ratio)
                elif aec is not None and mic_idx is not None:
                    # AEC is active but still learning the echo path (first ~1-2 s of agent speech):
                    # the raw mic is mostly echo, so no voice barge-in yet (hardware button still works).
                    conf = 0.0
                    is_barge = False
                else:
                    conf = float(self.vad.voice_confidence(pcm)[0]) if len(pcm) >= 512 else 0.0
                    is_queue_empty = False
                    if self.transport and hasattr(self.transport, "output"):
                        is_queue_empty = getattr(self.transport.output, "playback_queue", None) is not None and self.transport.output.playback_queue.empty()
                    # Legacy rule (AEC off / not converged yet): pause barge-in or loud close speech
                    is_pause_barge_in = is_queue_empty and (conf >= 0.60 and rms >= 0.03)
                    is_loud_barge_in = (conf >= self.echo_vad_threshold and rms >= self.barge_in_rms_threshold)
                    is_barge = is_pause_barge_in or is_loud_barge_in

                if probe_mode:
                    # echo-contaminated frames are zeroed so Gemini never hears the agent's own voice
                    self._bc_buffer.append(frame if evaluable else InputAudioRawFrame(
                        audio=b"\x00" * len(frame.audio), sample_rate=frame.sample_rate, num_channels=frame.num_channels))
                    if evaluable:
                        self._bc_win.append(1 if is_barge else 0)
                    self.barge_in_streak = sum(self._bc_win)
                else:
                    self._bc_buffer.append(send_frame)
                    self._bc_win.append(1 if is_barge else 0)
                    self.barge_in_streak = sum(self._bc_win)  # voiced frames in the last BARGE_IN_MIN_MS
                    if is_barge:
                        self._bc_peak = max(self._bc_peak, self.barge_in_streak)
                    elif self.barge_in_streak == 0:
                        if self._bc_peak >= 3:
                            print(f"💬 [BACKCHANNEL]: ignored short sound over agent (~{self._bc_peak*32}ms voiced < {self._bc_need*32}ms needed)")
                        self._bc_peak = 0

                if probe_mode:
                    # Echo-free evidence: >= 10 voiced frames (~320 ms) accumulated while the agent is held. Gaps between
                    # phrases are allowed; a 'yeah'/'uh-huh' (<~9 voiced frames) is still ignored.
                    confirm = self._pr_state == "hold" and self._pr_total >= 10
                else:
                    confirm = len(self._bc_win) >= self.barge_in_min_frames and self.barge_in_streak >= self._bc_need  # sustained speech (BARGE_IN_MIN_MS)
                if confirm:
                    now = time.perf_counter()
                    print(f"⚡ [VOICE BARGE-IN]: User voice interrupted speaker playback! (conf={conf:.2f}, rms={rms:.3f})")
                    if probe_mode:
                        # always-on context so a false barge-in can be diagnosed from the log alone
                        print(f"   [PROBE CONTEXT]: state={self._pr_state} echo_est={echo_est:.3f} "
                              f"agent_talking_for={self._pb_frames*0.032:.1f}s voiced_history(evaluable frames)="
                              f"{''.join(str(v) for v in self._bc_win)}", flush=True)
                    self.barge_in_streak = 0
                    self._bc_win.clear()
                    self._bc_peak = 0
                    self._pr_state = "none"
                    self.onset_streak = 0
                    self.user_speaking = True
                    self.speech_start_ts = now - 0.064
                    self.last_speech_ts = now

                    # Instantly stop speaker playback
                    if self.transport:
                        self.transport.cancel_playback_immediately()

                    if self.benchmark:
                        self.benchmark.start_new_turn()
                    if self.hw_mgr:
                        try:
                            self.hw_mgr.on_user_speaking(rms=rms)
                        except Exception:
                            pass
                    if getattr(self, "publisher", None):
                        self.publisher.set_state("listening", rms=rms)

                    # Notify Gemini of interruption and begin new user turn
                    await self.push_frame(InterruptionFrame(), direction)
                    await self.push_frame(UserStartedSpeakingFrame(), direction)
                    # deliver the buffered speech (ends with the current frame) so the first words survive
                    while self._bc_buffer:
                        await self.push_frame(self._bc_buffer.popleft(), direction)
                        await asyncio.sleep(0.002)
                    return

                if probe_mode:
                    self._probe_step(evaluable)

                # Normal echo gate: zero out audio sent to Gemini during speaker playback
                silent_pcm = b"\x00" * len(frame.audio)
                silent_frame = InputAudioRawFrame(
                    audio=silent_pcm,
                    sample_rate=frame.sample_rate,
                    num_channels=frame.num_channels
                )
                await self.push_frame(silent_frame, direction)
                return

            # -----------------------------------------------------------------
            # 2. SPEAKER IDLE (Agent is quiet, listening to user)
            # -----------------------------------------------------------------
            self.barge_in_streak = 0
            self._bc_buffer.clear()
            self._bc_win.clear()
            self._bc_peak = 0
            self._pb_frames = 0
            if self._pr_state != "none":
                self._probe_release("speaker idle")
            self._pr_since = 0
            if self.transport and hasattr(self.transport, "input") and hasattr(self.transport.input, "barge_in_active"):
                self.transport.input.barge_in_active = False

            # Query voice confidence from Silero VAD
            conf = float(self.vad.voice_confidence(pcm)[0]) if len(pcm) >= 512 else 0.0
            vad_state = await self.vad.analyze_audio(pcm)

            # Robust speech detection: neural voice confidence or state machine
            # Guarded by rms >= 0.015 (calibrated in tests/calibrate_vad_splicing.py)
            is_speech = (conf >= VAD_CONFIDENCE_THRESHOLD or vad_state == VADState.SPEAKING) and rms >= 0.015
            now = time.perf_counter()

            # Track onset streak (require 2 consecutive frames ~64ms of speech to confirm voice onset)
            if is_speech:
                self.onset_streak += 1
            else:
                self.onset_streak = 0

            # USER SPEECH ONSET
            if not self.user_speaking:
                if (self.onset_streak >= 2) or (vad_state == VADState.STARTING and is_speech):
                    if self._watchdog_task and not self._watchdog_task.done():
                        self._watchdog_task.cancel()

                    self.user_speaking = True
                    self.speech_start_ts = now - 0.064
                    self.last_speech_ts = now
                    self.onset_streak = 0
                    print(f"🎙️  [LOCAL VAD]: User started speaking at {now:.6f}")
                    if self.benchmark:
                        self.benchmark.start_new_turn()
                    if self.hw_mgr:
                        try:
                            self.hw_mgr.on_user_speaking(rms=rms)
                        except Exception:
                            pass

                    if TRACE:
                        self._tr_pre = len(self.preroll_buffer)
                        self._tr_in = 1
                        self._tr_out = len(self.preroll_buffer) + 1
                        self._tr_onset_wall = now
                        self._tr_max_gap = 0.0
                        print(f"[TRACE VAD] onset: preroll={self._tr_pre} frames ({self._tr_pre*32}ms) rms={rms:.3f} conf={conf:.2f} state={vad_state.name}", flush=True)

                    # 1. Inform downstream / Gemini that speech has started
                    await self.push_frame(UserStartedSpeakingFrame(), direction)

                    # 2. Sequentially deliver all pre-roll frames (~320ms)
                    # Smooth 2ms cadence ensures crisp speech onset without event-loop starvation
                    while self.preroll_buffer:
                        buffered_frame = self.preroll_buffer.popleft()
                        await self.push_frame(buffered_frame, direction)
                        await asyncio.sleep(0.002)

                    # 3. Deliver current frame
                    await self.push_frame(frame, direction)
                else:
                    # Still idle: store in rolling pre-roll buffer, do NOT stream room fan noise downstream
                    self.preroll_buffer.append(frame)
                return

            else:
                # USER IS SPEAKING
                # Update last speech timestamp during active voice OR while Silero analyzer is in STOPPING hysteresis
                if is_speech or (vad_state == VADState.STOPPING and rms >= 0.015):
                    self.last_speech_ts = now
                if rms > 0.015 and self.hw_mgr:
                    try:
                        self.hw_mgr.on_user_speaking(rms=rms)
                    except Exception:
                        pass

                if TRACE:
                    self._tr_in += 1
                silence_dur = now - self.last_speech_ts
                voice_dur = self.last_speech_ts - self.speech_start_ts

                # USER SPEECH END (Conversational turn detection matched to calibrate_vad_splicing.py)
                # Conclude turn after 550ms of quiet silence (allows natural inter-word breathing), or at silence_threshold (750ms)
                silence_threshold = (VAD_STOP_SPEECH_MS / 1000.0) if VAD_STOP_SPEECH_MS else 0.75
                is_end_of_speech = (vad_state == VADState.QUIET and silence_dur >= 0.55) or (silence_dur >= silence_threshold)
                if is_end_of_speech:
                    if TRACE:
                        _why = "QUIET>=0.55s" if (vad_state == VADState.QUIET and silence_dur >= 0.55) else f"silence>={silence_threshold:.2f}s"
                        print(f"[TRACE VAD] END ({_why}): in={self._tr_in} frames ({self._tr_in*32}ms audio) out={self._tr_out} frames ({self._tr_out*32}ms) "
                              f"wall_since_onset={(now-self._tr_onset_wall)*1000:.0f}ms voice={voice_dur:.2f}s silence={silence_dur:.2f}s "
                              f"max_arrival_gap={self._tr_max_gap*1000:.0f}ms (end frame itself is not forwarded)", flush=True)
                    self.user_speaking = False
                    self.onset_streak = 0
                    self.preroll_buffer.clear()

                    if voice_dur >= 0.20:
                        # Genuine user utterance concluded!
                        t_eos = self.last_speech_ts
                        print(f"🎙️  [LOCAL VAD]: End of user speech detected at {t_eos:.6f} (voice: {voice_dur:.2f}s, silence: {silence_dur:.2f}s)")

                        if self.benchmark:
                            self.benchmark.record_user_speech_end(t_eos)

                        if self.hw_mgr:
                            try:
                                self.hw_mgr.on_thinking("Processing response...")
                            except Exception:
                                pass

                        await self.push_frame(UserStoppedSpeakingFrame(), direction)

                        # Watchdog safety timer: prevents LCD/subsystems from hanging on 'THINKING'
                        # if Gemini Live remains silent or takes too long to infer.
                        async def _thinking_watchdog():
                            await asyncio.sleep(4.5)
                            is_playing = self.transport.is_playing if self.transport else False
                            if not is_playing and self.hw_mgr:
                                try:
                                    self.hw_mgr.on_idle()
                                except Exception:
                                    pass

                        if self._watchdog_task and not self._watchdog_task.done():
                            self._watchdog_task.cancel()
                        self._watchdog_task = asyncio.create_task(_thinking_watchdog())
                    else:
                        # Spurious transient noise (<0.20s voice) - cancel quietly
                        if self.hw_mgr:
                            try:
                                self.hw_mgr.on_idle()
                            except Exception:
                                pass
                    return

                if TRACE:
                    self._tr_out += 1
                await self.push_frame(frame, direction)
        else:
            await self.push_frame(frame, direction)

