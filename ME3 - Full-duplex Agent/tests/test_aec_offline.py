#!/usr/bin/env python3
"""
Offline check of agent/aec.py with a SIMULATED echo path (no hardware needed):
speaker signal -> 80 ms delay + decaying reverb + mild speaker nonlinearity -> mic,
plus sensor noise, plus a "user" talking over the agent in the middle.

Reports ERLE (how many dB of echo removed) and how much of the user's voice survives.
Run: /home/misael/AI_231_venv/bin/python tests/test_aec_offline.py
"""
import sys
import wave
from pathlib import Path

import numpy as np
from scipy.signal import fftconvolve

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from agent.aec import EchoCanceller, PlaybackReference, BLOCK  # noqa: E402

REC = BASE_DIR / "tests" / "recordings"
FS = 16000


def load(name):
    w = wave.open(str(REC / name))
    return np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0


def loop_to(x, n, shift=0):
    x = np.roll(x, shift)
    return np.tile(x, int(np.ceil(n / len(x))))[:n]


def rms(x):
    return float(np.sqrt(np.mean(x ** 2)) + 1e-12)


def run(delay_ms, gain, nonlinear, seed=0, mu=0.6, partitions=8):
    rng = np.random.default_rng(seed)
    n = FS * 20
    agent = loop_to(load("user_voice_16k_processed.wav"), n)
    agent = agent / rms(agent) * 0.10                    # speaker signal level
    user = loop_to(load("live_splice_turn_4.wav"), n, shift=12345)
    user = user / rms(user) * 0.10                       # near-end voice level

    ir = np.zeros(int(0.12 * FS))
    ir[0] = 1.0
    t = np.arange(len(ir)) / FS
    ir += rng.normal(0, 0.15, len(ir)) * np.exp(-t / 0.03)
    ir = np.concatenate((np.zeros(int(delay_ms / 1000 * FS)), ir)) * gain
    k = float(nonlinear)  # 0 = linear; 1.5 = mild; 3 = harsh speaker distortion
    spk = np.tanh(k * agent) / k if k else agent
    echo = fftconvolve(spk, ir)[:n].astype(np.float32)

    near = np.zeros(n, np.float32)
    s, e = 10 * FS, 14 * FS
    near[s:e] = user[s:e]                                # double-talk window 10..14 s
    mic = echo + near + rng.normal(0, 0.003, n).astype(np.float32)

    ref = PlaybackReference()
    aec = EchoCanceller(ref, partitions=partitions, mu=mu)
    res = np.zeros(n, np.float32)
    for b in range(n // BLOCK):
        sl = slice(b * BLOCK, (b + 1) * BLOCK)
        ref.write(agent[sl])                             # output callback writes first
        res[sl] = aec.process(b, mic[sl])

    seg = lambda a, lo, hi: a[int(lo * FS):int(hi * FS)]
    erle_early = 20 * np.log10(rms(seg(mic, 0, 2)) / rms(seg(res, 0, 2)))
    erle_conv = 20 * np.log10(rms(seg(echo, 5, 10)) / rms(seg(res, 5, 10) - seg(near, 5, 10)))
    # during double talk: residual should look like the near-end voice, not the echo
    r_near = float(np.corrcoef(seg(res, 10, 14), seg(near, 10, 14))[0, 1])
    r_echo_raw = float(np.corrcoef(seg(mic, 10, 14), seg(near, 10, 14))[0, 1])
    leak_dt = 20 * np.log10(rms(seg(res, 10, 14) - seg(near, 10, 14)) / rms(seg(echo, 10, 14)))
    return erle_early, erle_conv, r_near, r_echo_raw, leak_dt, aec.peak_delay_ms(), aec.converged


if __name__ == "__main__":
    print(f"{'delay':>6} {'gain':>5} {'nonlin':>6} | {'ERLE 0-2s':>9} {'ERLE 5-10s':>10} "
          f"{'corr(res,user)':>14} {'corr(mic,user)':>14} {'echo left in DT':>15} {'peak ms':>7} conv")
    ok = True
    for delay, gain, nl, must in [(40, 0.5, 0, True), (80, 0.5, 1.5, True), (120, 1.0, 1.5, True),
                                  (160, 0.8, 1.5, True), (80, 0.5, 3.0, False)]:
        a, b, c, d, f, pk, conv = run(delay, gain, nl)
        print(f"{delay:>6} {gain:>5} {nl:>6} | {a:>8.1f}dB {b:>9.1f}dB {c:>14.2f} {d:>14.2f} "
              f"{f:>13.1f}dB {pk:>7.0f} {conv}" + ("" if must else "   (harsh, informational)"))
        if must:
            ok &= (b >= 12.0) and conv and (c > d)
    print("PASS" if ok else "FAIL (see numbers above)")
    sys.exit(0 if ok else 1)
