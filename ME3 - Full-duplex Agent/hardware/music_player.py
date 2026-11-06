#!/usr/bin/env python3
"""
Dedicated Background Music Player for Full-Duplex Agent
Plays MP3/WAV tracks from media/ directory.
Integrates with GMarkAudioSystem's ALSA sounddevice stream mixer:
- Native 48,000 Hz, 2-channel float32 audio
- Automatic Ducking: Lowers volume to 15% when Gemini is speaking
- Instant Pause, Resume, Stop, Next, and Volume adjustment
- Full thread-safe operation
"""

import os
import glob
import threading
from pathlib import Path
import numpy as np
import soundfile as sf
from config import MUSIC_DIR, MUSIC_MAX_GAIN, MUSIC_DEFAULT_VOLUME_PCT, MUSIC_REF_RMS


class MusicPlayer:
    """Manages background music library and stream mixing chunks."""
    def __init__(self, music_dir=MUSIC_DIR):
        self.music_dir = Path(music_dir)
        self.music_files = []
        self.current_idx = 0
        self.current_audio = None  # 2D float32 array (N, 2)
        self.current_title = "None"
        self.play_pos = 0
        self.is_playing = False
        # User-facing volume fraction (0..1). Effective gain is quadratic and loudness-normalised so music stays
        # quiet enough for voice commands to be understood over it (see config.MUSIC_MAX_GAIN notes).
        self.volume = MUSIC_DEFAULT_VOLUME_PCT / 100.0
        self.norm = 1.0  # per-track loudness normalisation
        self.ducked_gain = 0.15  # Ducking factor when assistant speaks (15% of current vol)
        self.lock = threading.Lock()

        self.refresh_library()

    def refresh_library(self):
        """Scans media/ for audio files in alphabetical order."""
        if not self.music_dir.exists():
            self.music_dir.mkdir(parents=True, exist_ok=True)
            self.music_files = []
            return

        patterns = ["*.mp3", "*.wav", "*.ogg", "*.flac"]
        found = []
        for pat in patterns:
            found.extend(self.music_dir.glob(pat))
        found.sort(key=lambda p: p.name)
        self.music_files = found
        if self.music_files:
            print(f"🎵 [MUSIC PLAYER]: Found {len(self.music_files)} tracks in {self.music_dir.name}/")

    def _load_track(self, index: int):
        """Loads and caches audio data for track at index."""
        if not self.music_files:
            return False
        self.current_idx = index % len(self.music_files)
        file_path = self.music_files[self.current_idx]
        try:
            data, sr = sf.read(str(file_path), dtype="float32")
            if data.ndim == 1:
                data = np.column_stack((data, data))
            elif data.ndim > 1 and data.shape[1] > 2:
                data = data[:, :2]

            # Resample to 48,000 Hz if needed
            if sr != 48000:
                import scipy.signal
                target_len = int(len(data) * 48000 / sr)
                data = scipy.signal.resample(data, target_len).astype(np.float32)

            self.current_audio = data
            self.current_title = file_path.stem
            track_rms = float(np.sqrt(np.mean(data[:: 8] ** 2))) or 1e-6
            self.norm = float(np.clip(MUSIC_REF_RMS / track_rms, 0.25, 2.0))  # louder tracks are turned down
            self.play_pos = 0
            print(f"🎵 [MUSIC PLAYER]: Loaded '{self.current_title}' ({len(data)/48000:.1f}s)")
            return True
        except Exception as e:
            print(f"❌ [MUSIC PLAYER ERROR]: Failed to load {file_path.name}: {e}")
            self.current_audio = None
            return False

    def play(self, track_name_or_idx=None) -> str:
        """Starts or resumes music playback."""
        with self.lock:
            if not self.music_files:
                self.refresh_library()
            if not self.music_files:
                return "No music files found in media library."

            if track_name_or_idx is not None:
                # Find by index or substring
                if isinstance(track_name_or_idx, int):
                    self._load_track(track_name_or_idx)
                else:
                    search = str(track_name_or_idx).lower()
                    match_idx = None
                    for i, p in enumerate(self.music_files):
                        if search in p.name.lower():
                            match_idx = i
                            break
                    if match_idx is not None:
                        self._load_track(match_idx)
                    elif self.current_audio is None:
                        self._load_track(0)
            elif self.current_audio is None:
                self._load_track(0)

            self.is_playing = True
            return self.current_title

    def pause(self) -> bool:
        """Pauses music playback without resetting position."""
        with self.lock:
            self.is_playing = False
            return True

    def stop(self) -> bool:
        """Stops music playback and rewinds to the beginning."""
        with self.lock:
            self.is_playing = False
            self.play_pos = 0
            return True

    def next_track(self) -> str:
        """Advances to next track and starts playback."""
        with self.lock:
            if not self.music_files:
                return "No tracks available"
            self.current_idx = (self.current_idx + 1) % len(self.music_files)
            self._load_track(self.current_idx)
            self.is_playing = True
            return self.current_title

    def set_volume(self, vol_0_100: int) -> int:
        """Sets music playback volume (0 to 100)."""
        with self.lock:
            clamped = max(0, min(100, int(vol_0_100)))
            self.volume = clamped / 100.0
            return clamped

    def adjust_volume(self, delta: int) -> int:
        """Adjusts music volume by delta."""
        with self.lock:
            current_pct = int(self.volume * 100)
            new_pct = max(0, min(100, current_pct + delta))
            self.volume = new_pct / 100.0
            return new_pct

    def get_track_list(self) -> list:
        """Returns filenames of available tracks."""
        with self.lock:
            if not self.music_files:
                self.refresh_library()
            return [p.name for p in self.music_files]

    def get_audio_chunk(self, num_frames: int, ducked: bool = False) -> np.ndarray | None:
        """
        Called by sounddevice output_callback to fetch the next slice of audio.
        Returns a (num_frames, 2) float32 numpy array or None.
        """
        with self.lock:
            if not self.is_playing or self.current_audio is None:
                return None

            total_len = len(self.current_audio)
            if self.play_pos >= total_len:
                # Song finished: advance to next song
                self.current_idx = (self.current_idx + 1) % len(self.music_files)
                if not self._load_track(self.current_idx):
                    self.is_playing = False
                    return None
                total_len = len(self.current_audio)

            end_pos = min(self.play_pos + num_frames, total_len)
            chunk = self.current_audio[self.play_pos:end_pos]
            needed = num_frames - len(chunk)
            self.play_pos = end_pos

            # If track looped during this frame
            if needed > 0 and self.play_pos >= total_len:
                self.current_idx = (self.current_idx + 1) % len(self.music_files)
                if self._load_track(self.current_idx) and self.current_audio is not None:
                    extra = self.current_audio[:needed]
                    chunk = np.vstack((chunk, extra))
                    self.play_pos = len(extra)
                else:
                    # Pad remaining with zeros
                    pad = np.zeros((needed, 2), dtype=np.float32)
                    chunk = np.vstack((chunk, pad))

            # Apply volume and ducking gain
            eff_vol = MUSIC_MAX_GAIN * (self.volume ** 2) * self.norm * (self.ducked_gain if ducked else 1.0)
            return chunk * eff_vol

