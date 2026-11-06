#!/usr/bin/env python3
"""
Spotify audio source for the HDMI music writer (option A): runs librespot (Spotify Connect receiver) with its `pipe`
backend and hands its 44.1 kHz S16 stereo output to HdmiMusicOutput's writer thread, which resamples it to 48 kHz and mixes
it exactly like local music (same volume, same ducking under Berry's voice). Nothing here runs in a PortAudio callback or
on the asyncio loop: read_chunk() is called only from the music-hdmi thread, and the pipe is read NON-blocking, on demand,
so librespot is back-pressured by the pipe (no unbounded read-ahead; pause reacts in ~100 ms).
Revert/disable: ME3_SPOTIFY_PLAY=0. librespot is the official Raspotify build unpacked into ~/.local/share/berry-librespot.
"""

import os
import time
import fcntl
import atexit
import signal
import ctypes
import subprocess
import numpy as np
from math import gcd
from pathlib import Path

FRAMES_OUT = 1920            # 40 ms @ 48 kHz (HdmiMusicOutput frame)
FR_IN = 1764                 # 40 ms @ 44.1 kHz
CTX = 147                    # context frames on each side of a block (147 -> 160 output frames exactly: 48000/44100 = 160/147)
BYTES_PER_FRAME = 4          # S16 stereo
IDLE_RESET_S = 0.5           # no data this long => drop the stale tail and the resampler context


def _die_with_parent():
    try:  # PR_SET_PDEATHSIG: librespot gets SIGTERM if the agent dies unexpectedly
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, int(signal.SIGTERM))
    except Exception:
        pass


class SpotifySource:
    def __init__(self, binary, cache_dir, device_name, log_path=None):
        self.binary = str(binary)
        self.cache_dir = str(cache_dir)
        self.device_name = device_name
        self.log_path = Path(log_path) if log_path else None
        self.proc = None
        self._fd = None
        self._buf = bytearray()
        self._prev = None
        self._last_data = 0.0
        self._last_start = 0.0
        self.chunks_out = 0      # audio blocks delivered so far (tools use it to confirm that sound really started)
        self._log = None
        atexit.register(self.stop)

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def ensure_running(self) -> bool:
        """Starts librespot if it is not running (at most one attempt per 5 s). Safe to call from any worker thread."""
        if self.alive:
            return True
        if time.time() - self._last_start < 5.0:
            return False
        self._last_start = time.time()
        try:
            if self.log_path:
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                self._log = open(self.log_path, "ab")
            cmd = [self.binary, "--name", self.device_name, "--cache", self.cache_dir, "--system-cache", self.cache_dir,
                   "--disable-audio-cache", "--disable-discovery", "--backend", "pipe", "--format", "S16",
                   "--bitrate", "160", "--volume-ctrl", "fixed", "--initial-volume", "100",
                   "--enable-volume-normalisation"]
            self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                         stderr=self._log or subprocess.DEVNULL, preexec_fn=_die_with_parent)
            self._fd = self.proc.stdout.fileno()
            os.set_blocking(self._fd, False)
            try:  # small pipe (~93 ms): librespot is throttled by us instead of reading far ahead
                fcntl.fcntl(self._fd, 1031, 16384)   # F_SETPIPE_SZ
            except Exception:
                pass
            self._buf = bytearray()
            self._prev = None
            print(f"🟢 [SPOTIFY]: librespot started as Connect device '{self.device_name}'.", flush=True)
            return True
        except Exception as e:
            print(f"⚠️ [SPOTIFY]: could not start librespot ({type(e).__name__}: {e})", flush=True)
            self.proc = None
            return False

    def _pull(self):
        """Non-blocking read of what is needed to keep ~2 blocks buffered (never more). Leaving the rest in the pipe is what
        throttles librespot to real time and keeps pause latency ~0.1 s; draining it all would make the buffer overflow."""
        want_total = (FR_IN * 2 + CTX) * BYTES_PER_FRAME
        while self._fd is not None and len(self._buf) < want_total:
            try:
                b = os.read(self._fd, want_total - len(self._buf))
            except (BlockingIOError, OSError):
                return
            if not b:      # EOF: librespot exited (ensure_running() restarts it on the next tool call)
                return
            self._buf += b
            self._last_data = time.time()

    def read_chunk(self):
        """float32 (FRAMES_OUT, 2) at 48 kHz, or None when Spotify is not producing audio. Writer thread only."""
        if self.proc is None:
            return None
        self._pull()
        need = (FR_IN + CTX) * BYTES_PER_FRAME
        if len(self._buf) < need:
            if self._buf and time.time() - self._last_data > IDLE_RESET_S:
                self._buf.clear()   # paused/ended: do not replay a stale fragment on resume
                self._prev = None
            return None
        cur = np.frombuffer(bytes(self._buf[:need]), dtype="<i2").reshape(-1, 2).astype(np.float32) / 32768.0
        del self._buf[:FR_IN * BYTES_PER_FRAME]
        prev = self._prev if self._prev is not None else np.zeros((CTX, 2), dtype=np.float32)
        self._prev = cur[FR_IN - CTX:FR_IN].copy()
        import scipy.signal
        out = scipy.signal.resample_poly(np.vstack((prev, cur)), 160, 147, axis=0).astype(np.float32)
        out = out[160:160 + FRAMES_OUT]      # drop the 147-frame left context (= 160 out); right context is look-ahead
        if len(out) != FRAMES_OUT:
            return None
        self.chunks_out += 1
        return out

    def stop(self):
        p, self.proc = self.proc, None
        if p is not None:
            try:
                p.terminate()
                p.wait(timeout=2)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        if self._log:
            try:
                self._log.close()
            except Exception:
                pass
            self._log = None


class SoloistSource:
    """Spotify playback through Spotify Soloist (official headless client). Soloist can only output to PipeWire/PulseAudio,
    so it plays into a VIRTUAL PipeWire sink (`berry_spotify`, no hardware, created at runtime with pw-cli) and `pw-record`
    captures that sink as raw 48 kHz S16 stereo, which HdmiMusicOutput mixes exactly like local music (same volume/ducking).
    Same interface as SpotifySource. The capture is clocked by PipeWire (not by us), so the buffer is kept short:
    overflow drops the oldest audio; digital silence (idle sink) is reported as None so it never masks local music."""

    SINK = "berry_spotify"
    SILENCE_PEAK = 12            # int16 peak below this over a whole block = silence (about -68 dBFS)
    BLOCK = FRAMES_OUT * BYTES_PER_FRAME

    def __init__(self, binary, data_dir, cache_dir, device_name, api_key, log_dir=None):
        self.binary, self.data_dir, self.cache_dir = str(binary), str(data_dir), str(cache_dir)
        self.device_name, self.api_key = device_name, api_key
        self.log_dir = Path(log_dir) if log_dir else None
        self.proc = None          # soloist
        self.cap = None           # pw-record
        self._fd = None
        self._buf = bytearray()
        self._last_start = 0.0
        self._logs = []
        self.chunks_out = 0
        atexit.register(self.stop)

    @property
    def alive(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def _log(self, name):
        if not self.log_dir:
            return subprocess.DEVNULL
        self.log_dir.mkdir(parents=True, exist_ok=True)
        f = open(self.log_dir / name, "ab")
        self._logs.append(f)
        return f

    def _ensure_sink(self) -> bool:
        """Creates the virtual sink if it is missing (runtime object, no config file; gone after a PipeWire restart)."""
        try:
            ls = subprocess.run(["pw-cli", "ls", "Node"], capture_output=True, text=True, timeout=5).stdout
            if f'"{self.SINK}"' in ls:
                return True
            subprocess.run(["pw-cli", "create-node", "adapter",
                            "{ factory.name=support.null-audio-sink node.name=" + self.SINK +
                            ' node.description="Berry Spotify (virtual)" media.class=Audio/Sink '
                            "object.linger=true audio.position=[FL,FR] }"],
                           capture_output=True, text=True, timeout=5)
            time.sleep(0.5)
            ls = subprocess.run(["pw-cli", "ls", "Node"], capture_output=True, text=True, timeout=5).stdout
            return f'"{self.SINK}"' in ls
        except Exception as e:
            print(f"⚠️ [SPOTIFY]: could not create the virtual sink ({type(e).__name__}: {e})", flush=True)
            return False

    def ensure_running(self) -> bool:
        """Starts the virtual sink, Soloist and the capture if needed (rate-limited to once per 5 s)."""
        if self.alive and self.cap is not None and self.cap.poll() is None:
            return True
        if time.time() - self._last_start < 5.0:
            return self.alive
        self._last_start = time.time()
        try:
            if not self._ensure_sink():
                return False
            if not self.alive:
                cmd = [self.binary, "--device-name", self.device_name, "--api-key", self.api_key,
                       "--pipewire-device", self.SINK, "--initial-volume", "100",
                       "--data-dir", self.data_dir, "--cache-dir", self.cache_dir]
                self.proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=self._log("soloist.log"),
                                             stderr=subprocess.STDOUT, preexec_fn=_die_with_parent)
                print(f"🟢 [SPOTIFY]: Soloist started as Connect device '{self.device_name}'.", flush=True)
            if self.cap is None or self.cap.poll() is not None:
                self.cap = subprocess.Popen(
                    ["pw-record", "-a", "--target", self.SINK, "-P", "{ stream.capture.sink=true }", "--rate", "48000",
                     "--channels", "2", "--format", "s16", "--latency", "40ms", "-"],
                    stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                    preexec_fn=_die_with_parent)
                self._fd = self.cap.stdout.fileno()
                os.set_blocking(self._fd, False)
                try:
                    fcntl.fcntl(self._fd, 1031, 16384)
                except Exception:
                    pass
                self._buf = bytearray()
            return self.alive
        except Exception as e:
            print(f"⚠️ [SPOTIFY]: could not start Soloist ({type(e).__name__}: {e})", flush=True)
            return False

    def read_chunk(self):
        """float32 (FRAMES_OUT, 2) at 48 kHz, or None (no audio / silence). Writer thread only."""
        if self._fd is None:
            return None
        while len(self._buf) < self.BLOCK * 3:
            try:
                b = os.read(self._fd, self.BLOCK * 3 - len(self._buf))
            except (BlockingIOError, OSError):
                break
            if not b:
                break
            self._buf += b
        if len(self._buf) > self.BLOCK * 6:           # capture is not paced by us: drop the oldest, keep ~80 ms
            del self._buf[:(len(self._buf) - self.BLOCK * 2) & ~3]
        if len(self._buf) < self.BLOCK:
            return None
        x = np.frombuffer(bytes(self._buf[:self.BLOCK]), dtype="<i2")
        del self._buf[:self.BLOCK]
        if int(np.abs(x).max()) < self.SILENCE_PEAK:
            return None
        self.chunks_out += 1
        return x.reshape(-1, 2).astype(np.float32) / 32768.0

    def stop(self):
        for attr in ("cap", "proc"):
            p = getattr(self, attr)
            setattr(self, attr, None)
            if p is not None:
                try:
                    p.terminate()
                    p.wait(timeout=2)
                except Exception:
                    try:
                        p.kill()
                    except Exception:
                        pass
        self._fd = None
        for f in self._logs:
            try:
                f.close()
            except Exception:
                pass
        self._logs = []
