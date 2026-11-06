#!/usr/bin/env python3
"""
Audio Device Test for G-MARK Micro Go (USB)
Detects input/output audio devices, captures a 2-second snippet, and plays it back.
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import sounddevice as sd
from config import AUDIO_DEVICE_NAME, AUDIO_SAMPLE_RATE

def find_device_index(name_substr, kind="input"):
    devices = sd.query_devices()
    for idx, d in enumerate(devices):
        if name_substr.lower() in d["name"].lower():
            if kind == "input" and d["max_input_channels"] > 0:
                return idx, d["name"]
            elif kind == "output" and d["max_output_channels"] > 0:
                return idx, d["name"]
    return None, None

def main():
    print("=" * 60)
    print("AUDIO HARDWARE DETECTION & LOOPBACK TEST")
    print("=" * 60)
    print("\nAll Available Audio Devices:")
    print(sd.query_devices())

    in_idx, in_name = find_device_index(AUDIO_DEVICE_NAME, "input")
    out_idx, out_name = find_device_index(AUDIO_DEVICE_NAME, "output")

    print(f"\nTarget USB Microphone: [{in_idx}] {in_name}")
    print(f"Target USB Speaker:    [{out_idx}] {out_name}")

    if in_idx is None or out_idx is None:
        print("❌ Could not locate USB Composite Device. Check USB connection.")
        sys.exit(1)

    HW_SR = 48000
    duration = 2.0  # seconds
    print(f"\n🎙️  Recording {duration}s from mic (device {in_idx}) at hardware {HW_SR} Hz...")
    recording_hw = sd.rec(
        int(duration * HW_SR),
        samplerate=HW_SR,
        channels=1,
        dtype="float32",
        device=in_idx
    )
    sd.wait()

    rms = np.sqrt(np.mean(recording_hw ** 2))
    peak = np.max(np.abs(recording_hw))
    print(f"Hardware audio recorded: RMS={rms:.4f}, Peak={peak:.4f}")

    # Downsample 48kHz -> 16kHz (3:1 decimation)
    audio_16k = recording_hw[::3]
    print(f"Downsampled to 16kHz: length {len(audio_16k)} samples (expected {int(duration*16000)})")

    # Upsample 16kHz -> 48kHz (1:3 repeat)
    audio_48k = np.repeat(audio_16k, 3, axis=0)
    # Duplicate mono to stereo for 2-channel speaker output
    stereo_audio = np.repeat(audio_48k, 2, axis=1)

    print(f"🔊 Playing back {duration}s to speaker (device {out_idx}) at {HW_SR} Hz...")
    sd.play(stereo_audio, samplerate=HW_SR, device=out_idx)
    sd.wait()

    print("✅ Audio loopback test complete!")

if __name__ == "__main__":
    main()
