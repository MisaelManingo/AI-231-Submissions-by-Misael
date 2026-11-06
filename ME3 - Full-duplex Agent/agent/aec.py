#!/usr/bin/env python3
"""
Software Acoustic Echo Canceller (pure numpy, no extra packages).

The G-MARK mic and speaker share one USB housing, so the agent's own voice (and
background music) leaks straight into the mic. We know exactly what was sent to the
speaker, so we learn the speaker->mic echo path with a partitioned-block
frequency-domain adaptive filter (PBFDAF / "MDF", NLMS-style) and subtract the
predicted echo from every mic block. What is left (the residual) is mostly the user.

Data flow:
  output callback --write()--> PlaybackReference (ring, mono 16 kHz, post-mix)
  VAD processor   --process(block_idx, mic_block)--> residual block

Alignment: mic block N is paired with reference samples [N*512, ...). Both streams
start at (almost) the same time on the same USB card, so any start offset just shows
up as extra echo delay, which the filter span (partitions * 32 ms) absorbs.
"""

import numpy as np

BLOCK = 512  # 32 ms @ 16 kHz, same as AUDIO_CHUNK_SIZE


class PlaybackReference:
    """Ring buffer of what was handed to the speaker (mono, 16 kHz, after music mix)."""

    def __init__(self, seconds: int = 8, rate: int = 16000):
        self.size = int(seconds * rate)
        self.buf = np.zeros(self.size, dtype=np.float32)
        self.written = 0  # absolute sample count

    def write(self, x: np.ndarray):
        """Called from the PortAudio output callback: keep it tiny (no locks, no FFT)."""
        n = len(x)
        if n == 0:
            return
        w = self.written % self.size
        end = w + n
        if end <= self.size:
            self.buf[w:end] = x
        else:
            k = self.size - w
            self.buf[w:] = x[:k]
            self.buf[:n - k] = x[k:]
        self.written += n  # published last so readers never see half-written data

    def get(self, start: int, n: int) -> np.ndarray:
        """Returns n samples from absolute index `start`; zeros where unavailable."""
        out = np.zeros(n, dtype=np.float32)
        written = self.written
        lo = max(start, 0, written - self.size)
        hi = min(start + n, written)
        if hi > lo:
            out[lo - start:hi - start] = self.buf[np.arange(lo, hi) % self.size]
        return out


class EchoCanceller:
    def __init__(self, ref: PlaybackReference, partitions: int = 8, mu: float = 0.6,
                 converged_erle_db: float = 6.0):
        self.ref = ref
        self.P = partitions
        self.mu = mu
        self.converged_erle_db = converged_erle_db
        B = BLOCK
        self.W = np.zeros((self.P, B + 1), dtype=np.complex128)   # filter partitions
        self.X = np.zeros((self.P, B + 1), dtype=np.complex128)   # ref spectra, newest first
        self.last_idx = None
        self.erle_db = 0.0     # smoothed echo return loss enhancement (dB), echo-only blocks
        self.ref_active = False
        self.last_echo_rms = 0.0  # level of the echo the filter predicted for the last block
        self._dt_hold = 0         # double-talk hangover (blocks during which adaptation stays frozen)
        self.near_end = False     # True while we believe the user is talking over the agent

    def _spec(self, idx: int):
        win = self.ref.get((idx - 1) * BLOCK, 2 * BLOCK)
        return np.fft.rfft(win), float(np.mean(win[BLOCK:] ** 2))

    @property
    def converged(self) -> bool:
        return self.erle_db >= self.converged_erle_db

    def peak_delay_ms(self) -> float:
        """Where the filter has put its energy: approximate speaker->mic delay."""
        e = np.sum(np.abs(self.W) ** 2, axis=1)
        return float(np.argmax(e) * BLOCK / 16.0) if e.max() > 0 else 0.0

    def process(self, idx: int, mic: np.ndarray) -> np.ndarray:
        """mic: float32 block of BLOCK samples (-1..1). Returns residual (echo removed)."""
        B = BLOCK
        if self.last_idx is None or idx != self.last_idx + 1:
            # first block, or mic blocks were skipped: rebuild reference history from the ring
            for p in range(self.P):
                self.X[p] = self._spec(idx - p)[0]
            _, ref_pow = self._spec(idx)
        else:
            self.X = np.roll(self.X, 1, axis=0)
            self.X[0], ref_pow = self._spec(idx)
        self.last_idx = idx

        Y = np.sum(self.W * self.X, axis=0)
        y = np.fft.irfft(Y, 2 * B)[B:].astype(np.float32)
        e = mic - y

        d_pow = float(np.mean(mic ** 2))
        e_pow = float(np.mean(e ** 2))
        y_pow = float(np.mean(y ** 2))
        self.last_echo_rms = y_pow ** 0.5
        hist_pow = float(np.sum(np.abs(self.X) ** 2))
        self.ref_active = hist_pow > 1e-3

        # DOUBLE-TALK GUARD: once converged we know how small the residual should be for echo alone.
        # A much bigger error means the user is talking; adapting then would teach the filter to
        # cancel the user's voice, so freeze adaptation (plus a short hangover).
        if self.converged:
            expected = y_pow * 10.0 ** (-self.erle_db / 10.0) + 1e-5
            if e_pow > 4.0 * expected:
                self._dt_hold = 10
        self.near_end = self._dt_hold > 0
        if self._dt_hold > 0:
            self._dt_hold -= 1

        if self.ref_active and not self.near_end:
            # Variable step size: when the error is mostly "echo we already predict" adapt fast;
            # when it is dominated by something we cannot predict (the user talking) adapt slowly.
            ratio = y_pow / (y_pow + e_pow + 1e-12)
            mu_eff = self.mu * (0.15 + 0.85 * ratio)
            E = np.fft.rfft(np.concatenate((np.zeros(B, dtype=np.float32), e)))
            norm = np.sum(np.abs(self.X) ** 2, axis=0) + 1e-6
            self.W += mu_eff * np.conj(self.X) * E / norm
            # gradient constraint: keep each partition causal (first B taps only)
            w = np.fft.irfft(self.W, 2 * B, axis=1)
            w[:, B:] = 0.0
            self.W = np.fft.rfft(w, axis=1)

            if ref_pow > 1e-5 and d_pow > 1e-6:
                erle = 10.0 * np.log10(d_pow / (e_pow + 1e-12))
                self.erle_db = 0.95 * self.erle_db + 0.05 * float(np.clip(erle, -10.0, 40.0))
        return e
