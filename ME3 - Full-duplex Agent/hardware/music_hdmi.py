#!/usr/bin/env python3
"""
Routes background music to the HDMI monitor speaker instead of mixing it into the USB speaker stream, so music can be
loud without the shared USB mic/speaker housing hearing it (agent voice stays on the USB speaker).

Why a subprocess: the Pi's HDMI ALSA device (hw:N,0) only accepts raw IEC958 frames, which PortAudio cannot open.
ALSA's `plughw:` plugin converts normal PCM, so a writer THREAD pipes int16 stereo PCM into `aplay`. Nothing here runs
in a PortAudio callback or on the asyncio loop.

Safety: `alive` is False whenever aplay is not running; GMarkAudioSystem then mixes music into the USB stream again
(the previous behaviour). Disable entirely with MUSIC_TO_HDMI=0. Override the device with MUSIC_HDMI_DEVICE.
Does NOT touch ALSA mixer / volume settings.
"""

import os
import re
import time
import fcntl
import threading
import subprocess
from pathlib import Path
import numpy as np
from config import MUSIC_MAX_GAIN

RATE = 48000
FRAMES = 1920  # 40 ms per write


def find_hdmi_alsa_device() -> str | None:
    """plughw device for the first *connected* HDMI port (HDMI-A-N <-> ALSA card vc4hdmi{N-1}). None if none."""
    override = os.getenv("MUSIC_HDMI_DEVICE", "").strip()
    if override:
        return override
    for status in sorted(Path("/sys/class/drm").glob("card*-HDMI-A-*/status")):
        try:
            if status.read_text().strip() != "connected":
                continue
        except Exception:
            continue
        m = re.search(r"HDMI-A-(\d+)", status.parent.name)
        if m:
            return f"plughw:CARD=vc4hdmi{int(m.group(1)) - 1},DEV=0"
    return None


def to_int16_stereo(chunk: np.ndarray, gain_scale: float) -> bytes:
    """float32 (N,2) -> clipped little-endian int16 bytes, scaled by gain_scale."""
    return (np.clip(chunk * gain_scale, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()


class HdmiMusicOutput:
    def __init__(self, music_player, is_ducked, gain_scale: float = 1.0):
        self.player = music_player
        self.is_ducked = is_ducked          # callable -> bool (duck while the agent speaks, as before)
        self.gain_scale = gain_scale        # relative to the player's own gain (MUSIC_HDMI_MAX_GAIN / MUSIC_MAX_GAIN)
        self.proc = None
        self._thread = None
        self._stop = threading.Event()
        self.device = None
        self.spotify = None                 # SpotifySource (hardware/spotify_source.py), attached by mcp_server/extras.py

    @property
    def alive(self) -> bool:
        p = self.proc
        return p is not None and p.poll() is None

    def _spawn(self) -> bool:
        try:
            self.proc = subprocess.Popen(
                ["aplay", "-q", "-D", self.device, "-t", "raw", "-f", "S16_LE", "-r", str(RATE), "-c", "2",
                 "-B", "150000", "-"],
                stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
            try:  # small pipe => pause/skip react within ~tens of ms instead of ~0.3 s (best effort)
                fcntl.fcntl(self.proc.stdin.fileno(), 1031, 4096)  # F_SETPIPE_SZ
            except Exception:
                pass
            time.sleep(0.3)
            return self.alive
        except Exception as e:
            print(f"🎵 [MUSIC HDMI]: could not start aplay ({e})", flush=True)
            return False

    def start(self) -> bool:
        self.device = find_hdmi_alsa_device()
        if not self.device:
            print("🎵 [MUSIC HDMI]: no connected HDMI output found; music stays on the USB speaker.", flush=True)
            return False
        if not self._spawn():
            print(f"🎵 [MUSIC HDMI]: {self.device} did not open; music stays on the USB speaker.", flush=True)
            return False
        self._thread = threading.Thread(target=self._run, name="music-hdmi", daemon=True)
        self._thread.start()
        self.player.hdmi = self   # lets MusicPlayer.play_oneshot() confirm the HDMI speaker is alive
        print(f"🎵 [MUSIC HDMI]: music routed to {self.device} (agent voice stays on USB).", flush=True)
        return True

    def _run(self):
        silence = bytes(FRAMES * 4)
        failed_at = 0.0
        while not self._stop.is_set():
            try:
                if not self.alive:
                    # aplay died (monitor asleep/unplugged?): music falls back to USB; retry every 5 s
                    if time.time() - failed_at > 5.0:
                        failed_at = time.time()
                        if self._spawn():
                            print("🎵 [MUSIC HDMI]: HDMI output recovered.", flush=True)
                    time.sleep(0.2)
                    continue
                data = silence
                sp_chunk = None
                sp = self.spotify
                # Spotify (piped librespot) is mixed like local music: same volume setting, same ducking under Berry's voice.
                # A generated sound clip (player._saved) has priority; Spotify then simply waits (back-pressured pipe).
                if sp is not None and getattr(self.player, "_saved", None) is None:
                    sp_chunk = sp.read_chunk()
                    if sp_chunk is not None:
                        eff = MUSIC_MAX_GAIN * (self.player.volume ** 2) * (self.player.ducked_gain if self.is_ducked() else 1.0)
                        data = to_int16_stereo(sp_chunk, eff * self.gain_scale)
                if sp_chunk is None and self.player.is_playing:
                    chunk = self.player.get_audio_chunk(FRAMES, ducked=bool(self.is_ducked()))
                    if chunk is not None:
                        data = to_int16_stereo(chunk, self.gain_scale)
                self.proc.stdin.write(data)   # blocks at aplay's pace; this thread only
                self.proc.stdin.flush()
            except (BrokenPipeError, ValueError, OSError):
                time.sleep(0.2)
            except Exception as e:
                print(f"🎵 [MUSIC HDMI]: writer error {type(e).__name__}: {e}", flush=True)
                time.sleep(0.5)

    def stop(self):
        self._stop.set()
        if self.spotify is not None:
            self.spotify.stop()
        p = self.proc
        if p is not None:
            try:
                p.stdin.close()
            except Exception:
                pass
            try:
                p.terminate()
            except Exception:
                pass
