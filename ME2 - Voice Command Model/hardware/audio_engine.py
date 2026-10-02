#!/usr/bin/env python3
"""
Unified Software Audio Mixer & Player for Raspberry Pi 5
- PyTorch-Free audio decoding (pure wave + miniaudio / ffmpeg fallback)
- Single sounddevice OutputStream with background software mixer
- Separate gains for music vs. chimes/TTS
- Ducking support during wake-word command listening
- Non-blocking playlist management (Play, Pause, Stop, Next, Volume)
"""

import os
import sys
import glob
import time
import wave
import queue
import threading
import subprocess
import numpy as np

try:
    import sounddevice as sd
except (ImportError, OSError):
    sd = None

from .config import (
    MUSIC_DIR, TTS_DIR, CHIME_WAKE_PATH, CHIME_END_PATH, ALARM_SOUND_PATH,
    AUDIO_SAMPLE_RATE, DEFAULT_MUSIC_VOLUME, DUCKED_MUSIC_VOLUME, MOCK_HARDWARE
)

def decode_audio_file(filepath, target_sr=AUDIO_SAMPLE_RATE):
    """
    Decodes a WAV or MP3 file into a 1D float32 NumPy array (-1.0 to 1.0)
    resampled to target_sr. 100% PyTorch-free.
    """
    if not os.path.exists(filepath):
        return np.zeros(0, dtype=np.float32)

    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".wav":
        try:
            with wave.open(filepath, "rb") as wf:
                channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                num_frames = wf.getnframes()
                raw_bytes = wf.readframes(num_frames)

            if sample_width == 2:
                audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            elif sample_width == 1:
                audio = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
            elif sample_width == 4:
                audio = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
            else:
                audio = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0

            if channels > 1:
                audio = audio.reshape(-1, channels).mean(axis=1)

            if framerate != target_sr and len(audio) > 0:
                target_len = int(len(audio) * target_sr / framerate)
                audio = np.interp(
                    np.linspace(0, len(audio) - 1, target_len),
                    np.arange(len(audio)),
                    audio
                ).astype(np.float32)

            return audio
        except Exception as e:
            print(f"[AUDIO DECODER ERROR]: Failed decoding WAV {filepath}: {e}")

    # For MP3 files:
    # 1. Try miniaudio if installed
    try:
        import miniaudio
        decoded = miniaudio.decode(filepath, nchannels=1, sample_rate=target_sr, dtypes=miniaudio.SampleFormat.FLOAT32)
        return np.frombuffer(decoded.samples, dtype=np.float32)
    except (ImportError, Exception):
        pass

    # 2. Try ffmpeg via subprocess
    try:
        cmd = [
            "ffmpeg", "-v", "quiet", "-i", filepath,
            "-f", "s16le", "-ac", "1", "-ar", str(target_sr), "-"
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        raw_out, _ = proc.communicate(timeout=10)
        if proc.returncode == 0 and len(raw_out) > 0:
            return np.frombuffer(raw_out, dtype=np.int16).astype(np.float32) / 32768.0
    except Exception as e:
        print(f"[AUDIO DECODER ERROR]: ffmpeg MP3 decoding failed for {filepath}: {e}")

    return np.zeros(0, dtype=np.float32)


class AudioEngine:
    def __init__(self, sample_rate=AUDIO_SAMPLE_RATE, mock=False):
        self.sr = sample_rate
        self.mock = mock or MOCK_HARDWARE or (sd is None)
        
        self.lock = threading.Lock()
        self.stream = None
        self.running = True

        # Music playback state
        self.music_files = []
        self.current_song_idx = 0
        self.music_audio = np.zeros(0, dtype=np.float32)
        self.music_pos = 0
        self.music_playing = False
        self.music_volume = DEFAULT_MUSIC_VOLUME  # 0 to 100
        self.is_ducked = False

        # SFX / TTS channel queue
        self.sfx_buffer = np.zeros(0, dtype=np.float32)
        self.sfx_pos = 0
        self.sfx_active = False
        self.sfx_finished_event = threading.Event()
        self.sfx_finished_event.set()

        # Alarm channel
        self.alarm_audio = np.zeros(0, dtype=np.float32)
        self.alarm_pos = 0
        self.alarm_active = False
        self.alarm_repeats_left = 0
        self.alarm_event = threading.Event()

        self._refresh_music_library()
        self._init_audio_stream()

    def _refresh_music_library(self):
        """Scans Music/ folder in filename order."""
        if not os.path.exists(MUSIC_DIR):
            os.makedirs(MUSIC_DIR, exist_ok=True)
        patterns = ["*.mp3", "*.wav", "*.ogg", "*.flac"]
        found = []
        for pat in patterns:
            found.extend(glob.glob(os.path.join(MUSIC_DIR, pat)))
        found.sort(key=lambda x: os.path.basename(x))
        self.music_files = found
        if self.music_files:
            print(f"[AUDIO ENGINE]: Found {len(self.music_files)} tracks in {MUSIC_DIR}")
        else:
            print(f"[AUDIO ENGINE]: No music files found in {MUSIC_DIR}. Add songs like '[01] Song.mp3'")

    def _init_audio_stream(self):
        if self.mock:
            print("[AUDIO ENGINE]: Running in MOCK mode (audio muted/logged).")
            return

        try:
            self.stream = sd.OutputStream(
                channels=1,
                samplerate=self.sr,
                blocksize=1024,
                callback=self._audio_callback
            )
            self.stream.start()
            print(f"[AUDIO ENGINE]: SoundDevice OutputStream initialized @ {self.sr} Hz.")
        except Exception as e:
            print(f"[AUDIO ENGINE WARNING]: Could not open audio output stream ({e}). Switching to mock.")
            self.mock = True

    def _audio_callback(self, outdata, frames, time_info, status):
        """Single mixer callback run by ALSA / sounddevice."""
        chunk = np.zeros(frames, dtype=np.float32)

        with self.lock:
            # 1. ALARM (Priority 1: overrides everything else)
            if self.alarm_active and len(self.alarm_audio) > 0:
                rem = len(self.alarm_audio) - self.alarm_pos
                to_copy = min(frames, rem)
                chunk[:to_copy] += self.alarm_audio[self.alarm_pos : self.alarm_pos + to_copy]
                self.alarm_pos += to_copy

                if self.alarm_pos >= len(self.alarm_audio):
                    self.alarm_repeats_left -= 1
                    self.alarm_pos = 0
                    if self.alarm_repeats_left <= 0:
                        self.alarm_active = False
                        self.alarm_event.set()

                # Alarm is active, mute music and sfx
                outdata[:] = np.clip(chunk, -1.0, 1.0)[:, np.newaxis]
                return

            # 2. SFX / TTS Channel
            if self.sfx_active and len(self.sfx_buffer) > 0:
                rem = len(self.sfx_buffer) - self.sfx_pos
                to_copy = min(frames, rem)
                chunk[:to_copy] += self.sfx_buffer[self.sfx_pos : self.sfx_pos + to_copy]
                self.sfx_pos += to_copy

                if self.sfx_pos >= len(self.sfx_buffer):
                    self.sfx_active = False
                    self.sfx_buffer = np.zeros(0, dtype=np.float32)
                    self.sfx_finished_event.set()

            # 3. Background Music Channel
            if self.music_playing and len(self.music_audio) > 0:
                gain = (DUCKED_MUSIC_VOLUME if self.is_ducked else self.music_volume) / 100.0
                rem = len(self.music_audio) - self.music_pos
                to_copy = min(frames, rem)
                chunk[:to_copy] += self.music_audio[self.music_pos : self.music_pos + to_copy] * gain
                self.music_pos += to_copy

                # Track finished: advance to next song automatically
                if self.music_pos >= len(self.music_audio):
                    self._load_next_track(auto_play=True)

        outdata[:] = np.clip(chunk, -1.0, 1.0)[:, np.newaxis]

    # -----------------------------------------------------
    # MUSIC CONTROLS
    # -----------------------------------------------------
    def play_music(self):
        """Plays current song or first song in Music/."""
        with self.lock:
            if not self.music_files:
                self._refresh_music_library()
            if not self.music_files:
                print("🎵 [MUSIC]: No music files available in Music/.")
                return

            if len(self.music_audio) == 0:
                filepath = self.music_files[self.current_song_idx]
                print(f"🎵 [MUSIC]: Loading track '{os.path.basename(filepath)}'...")
                self.music_audio = decode_audio_file(filepath, self.sr)
                self.music_pos = 0

            self.music_playing = True
            track_name = os.path.basename(self.music_files[self.current_song_idx])
            print(f"🎵 [MUSIC]: Playing '{track_name}' (Vol: {self.music_volume}%)")

    def pause_music(self):
        """Pauses music, preserving playback position."""
        with self.lock:
            self.music_playing = False
            print("🎵 [MUSIC]: Music paused.")

    def stop_music(self):
        """Stops music, resetting playback to the start of current song."""
        with self.lock:
            self.music_playing = False
            self.music_pos = 0
            print("🎵 [MUSIC]: Music stopped (rewound to start).")

    def next_track(self):
        """Skips to next song in filename order (wraps around)."""
        with self.lock:
            self._load_next_track(auto_play=self.music_playing)

    def _load_next_track(self, auto_play=True):
        if not self.music_files:
            return
        self.current_song_idx = (self.current_song_idx + 1) % len(self.music_files)
        filepath = self.music_files[self.current_song_idx]
        print(f"⏭️  [MUSIC]: Next track -> '{os.path.basename(filepath)}'")
        self.music_audio = decode_audio_file(filepath, self.sr)
        self.music_pos = 0
        self.music_playing = auto_play

    def volume_up(self, step=10):
        with self.lock:
            self.music_volume = min(100, self.music_volume + step)
            print(f"🔊 [MUSIC VOLUME]: {self.music_volume}%")

    def volume_down(self, step=10):
        with self.lock:
            self.music_volume = max(0, self.music_volume - step)
            print(f"🔉 [MUSIC VOLUME]: {self.music_volume}%")

    def duck(self):
        """Ducks music during command listening."""
        with self.lock:
            self.is_ducked = True

    def unduck(self):
        """Restores music volume after command & TTS finish."""
        with self.lock:
            self.is_ducked = False

    # -----------------------------------------------------
    # SFX & TTS PLAYBACK
    # -----------------------------------------------------
    def play_sfx(self, filepath, wait=False):
        """Plays a chime or TTS clip. Optionally blocks until finished."""
        if not os.path.exists(filepath):
            if wait:
                time.sleep(0.3)
            return

        audio = decode_audio_file(filepath, self.sr)
        if len(audio) == 0:
            return

        if self.mock:
            print(f"[MOCK SFX]: Playing '{os.path.basename(filepath)}'")
            if wait:
                time.sleep(len(audio) / self.sr)
            return

        with self.lock:
            self.sfx_buffer = audio
            self.sfx_pos = 0
            self.sfx_active = True
            self.sfx_finished_event.clear()

        if wait:
            self.sfx_finished_event.wait(timeout=10.0)

    def play_tts_sequence(self, filepaths, wait=True):
        """Concatenates multiple TTS audio clips and plays them seamlessly."""
        clips = []
        for fp in filepaths:
            if os.path.exists(fp):
                c = decode_audio_file(fp, self.sr)
                if len(c) > 0:
                    clips.append(c)

        if not clips:
            return

        merged = np.concatenate(clips)
        if self.mock:
            print(f"[MOCK TTS SEQ]: Playing sequence of {len(filepaths)} clips")
            if wait:
                time.sleep(len(merged) / self.sr)
            return

        with self.lock:
            self.sfx_buffer = merged
            self.sfx_pos = 0
            self.sfx_active = True
            self.sfx_finished_event.clear()

        if wait:
            self.sfx_finished_event.wait(timeout=15.0)

    # -----------------------------------------------------
    # ALARM PLAYBACK
    # -----------------------------------------------------
    def trigger_alarm(self, repeats=10):
        """Plays Alarm.mp3 in a loop up to 10 times or until stopped."""
        print(f"🚨 [ALARM TRIGGERED]: Playing '{os.path.basename(ALARM_SOUND_PATH)}' ({repeats}x)")
        audio = decode_audio_file(ALARM_SOUND_PATH, self.sr)
        if len(audio) == 0:
            # Generate synth alarm tone fallback if file is missing
            t = np.linspace(0, 1.0, self.sr)
            audio = (0.5 * np.sin(2 * np.pi * 880 * t)).astype(np.float32)

        with self.lock:
            self.alarm_audio = audio
            self.alarm_pos = 0
            self.alarm_repeats_left = repeats
            self.alarm_active = True
            self.alarm_event.clear()

    def stop_alarm(self):
        """Stops alarm immediately."""
        with self.lock:
            if self.alarm_active:
                self.alarm_active = False
                self.alarm_repeats_left = 0
                self.alarm_event.set()
                print("🛑 [ALARM]: Alarm silenced.")

    def close(self):
        self.running = False
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
            except Exception:
                pass

