#!/usr/bin/env python3
"""
Audio Playback Diagnostic Utility
Plays back saved recordings from tests/recordings/ through the G-MARK Micro Go USB speaker
so you can hear exactly what audio was delivered to Gemini Live.

Usage:
    python tests/play_recording.py              # Plays latest recorded turn
    python tests/play_recording.py 1            # Plays live_sent_turn_1.wav
    python tests/play_recording.py --list       # Lists all saved recordings
"""

import sys
import argparse
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import sounddevice as sd
import soundfile as sf
from config import AUDIO_DEVICE_NAME


def get_speaker_index():
    devices = sd.query_devices()
    for idx, d in enumerate(devices):
        if AUDIO_DEVICE_NAME.lower() in d.get("name", "").lower():
            if d.get("max_output_channels", 0) > 0:
                return idx
    return sd.default.device[1]


def main():
    parser = argparse.ArgumentParser(description="Play back recorded agent audio slices")
    parser.add_argument("turn", nargs="?", default=None, help="Turn number (e.g. 1, 2) or filename")
    parser.add_argument("--list", action="store_true", help="List all saved recordings")
    args = parser.parse_args()

    rec_dir = BASE_DIR / "tests" / "recordings"
    if not rec_dir.exists():
        print(f"Recordings directory not found: {rec_dir}")
        return

    files = sorted(rec_dir.glob("*.wav"), key=lambda p: p.stat().st_mtime)

    if args.list:
        print("=" * 65)
        print("SAVED AUDIO RECORDINGS IN tests/recordings/:")
        print("=" * 65)
        for idx, f in enumerate(files, 1):
            info = sf.info(str(f))
            print(f"[{idx:2d}] {f.name:32s} | {info.duration:5.2f}s | {info.samplerate}Hz | {info.channels}ch")
        print("=" * 65)
        return

    target_file = None
    if args.turn is not None:
        if args.turn.isdigit():
            cand = rec_dir / f"live_sent_turn_{args.turn}.wav"
            if cand.exists():
                target_file = cand
            else:
                cand2 = rec_dir / f"live_splice_turn_{args.turn}.wav"
                if cand2.exists():
                    target_file = cand2
        else:
            target_file = rec_dir / args.turn if not Path(args.turn).is_file() else Path(args.turn)

    if target_file is None or not target_file.exists():
        # Default to latest recorded turn
        sent_files = sorted(rec_dir.glob("live_sent_turn_*.wav"), key=lambda p: p.stat().st_mtime)
        if sent_files:
            target_file = sent_files[-1]
        elif files:
            target_file = files[-1]
        else:
            print("No audio recordings found in tests/recordings/.")
            return

    print("=" * 65)
    print(f"PLAYING AUDIO RECORDING: {target_file.name}")
    print("=" * 65)

    data, sr = sf.read(str(target_file), dtype="float32")
    peak = float(np.max(np.abs(data)))
    rms = float(np.sqrt(np.mean(data ** 2)))
    print(f"File: {target_file}")
    print(f"Duration: {len(data)/sr:.2f} seconds | Peak: {peak:.3f} | RMS: {rms:.3f} | Sample Rate: {sr} Hz")

    out_idx = get_speaker_index()
    print(f"Output Speaker Device: [{out_idx}] '{sd.query_devices(out_idx)['name']}'")

    # Upsample to 48 kHz stereo for G-MARK speaker if 16 kHz
    if sr == 16000:
        audio_48k = np.repeat(data, 3)
    else:
        audio_48k = data

    if audio_48k.ndim == 1:
        stereo_48k = np.column_stack((audio_48k, audio_48k))
    else:
        stereo_48k = audio_48k

    print("\n🔊 Playing through speaker now...")
    try:
        sd.play(stereo_48k, samplerate=48000, device=out_idx)
        sd.wait()
        print("✅ Playback finished.")
    except Exception as e:
        print(f"❌ Playback error: {e}")


if __name__ == "__main__":
    main()

