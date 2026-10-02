#!/usr/bin/env python3
"""
make_actions_zip.py

Creates ME2-actions-quickstart.zip containing all models, data, hardware drivers,
audio assets, and standalone runner for the complete voice assistant with physical actions.
"""

import os
import sys
import glob
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ZIP_NAME = "ME2-actions-quickstart.zip"
ZIP_OUT = os.path.join(BASE_DIR, ZIP_NAME)

FILES_TO_PACK = [
    # Core Pipeline & Action Scripts
    ("demo_rpi5.py", "demo_rpi5.py"),
    ("demo_rpi5_action.py", "demo_rpi5_action.py"),
    ("simulate_demo.py", "simulate_demo.py"),
    ("test_all_actions.py", "test_all_actions.py"),
    ("generate_tts_assets.py", "generate_tts_assets.py"),
    ("requirements-pi.txt", "requirements-pi.txt"),
    ("QUICKSTART.md", "QUICKSTART.md"),
    # Audio Chimes & Fallback Alarm
    ("chime_wake.wav", "chime_wake.wav"),
    ("chime_end.wav", "chime_end.wav"),
    ("Alarm.mp3", "Alarm.mp3"),
    # Required data files
    ("data/labels_32.json", "data/labels_32.json"),
]

EXPORTS_ASSETS = [
    ("exports/labels_32.json", "exports/labels_32.json"),
    ("exports/mel_filters_40.npy", "exports/mel_filters_40.npy"),
    ("exports/hann_window_400.npy", "exports/hann_window_400.npy"),
    ("exports/bcresnet_32class_int8.onnx", "exports/bcresnet_32class_int8.onnx"),
    ("exports/wakeword_int8.onnx", "exports/wakeword_int8.onnx"),
]


def resolve_asset_path(base_dir, src_rel):
    primary = os.path.join(base_dir, src_rel)
    if os.path.exists(primary):
        return primary
    if src_rel.startswith("exports/"):
        fname = os.path.basename(src_rel)
        v4_path = os.path.join(base_dir, "exports/v4_32class", fname)
        if os.path.exists(v4_path):
            return v4_path
    return primary


def main():
    print("=" * 65)
    print("📦 BUILDING STANDALONE ME2-actions-quickstart.zip")
    print("=" * 65)

    if os.path.exists(ZIP_OUT):
        os.remove(ZIP_OUT)

    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        # 1. Base files
        for src_rel, dst_rel in FILES_TO_PACK:
            src = resolve_asset_path(BASE_DIR, src_rel)
            if os.path.exists(src):
                zf.write(src, arcname=dst_rel)
                print(f"  + Added: {dst_rel}")
            else:
                print(f"  ! Warning: {src_rel} not found!")

        # 2. Exports assets
        for src_rel, dst_rel in EXPORTS_ASSETS:
            src = resolve_asset_path(BASE_DIR, src_rel)
            if os.path.exists(src):
                zf.write(src, arcname=dst_rel)
                print(f"  + Added: {dst_rel} ({os.path.getsize(src) / 1024:.1f} KB)")
            else:
                print(f"  ! Warning: {src_rel} not found!")

        # 3. Hardware package
        hw_dir = os.path.join(BASE_DIR, "hardware")
        if os.path.exists(hw_dir):
            for root, dirs, files in os.walk(hw_dir):
                if "__pycache__" in root:
                    continue
                for f in files:
                    if f.endswith(".py"):
                        full_p = os.path.join(root, f)
                        rel_p = os.path.relpath(full_p, BASE_DIR)
                        zf.write(full_p, arcname=rel_p)
                        print(f"  + Added: {rel_p}")

        # 4. Music folder
        music_dir = os.path.join(BASE_DIR, "Music")
        if os.path.exists(music_dir):
            for root, dirs, files in os.walk(music_dir):
                for f in files:
                    full_p = os.path.join(root, f)
                    rel_p = os.path.relpath(full_p, BASE_DIR)
                    zf.write(full_p, arcname=rel_p)
                    print(f"  + Added: {rel_p}")

        # 5. TTS clips
        tts_dir = os.path.join(BASE_DIR, "tts")
        if os.path.exists(tts_dir):
            for root, dirs, files in os.walk(tts_dir):
                for f in files:
                    if f.endswith(".wav"):
                        full_p = os.path.join(root, f)
                        rel_p = os.path.relpath(full_p, BASE_DIR)
                        zf.write(full_p, arcname=rel_p)
            print(f"  + Added: {len(os.listdir(tts_dir))} TTS audio clips from tts/")

    sz_mb = os.path.getsize(ZIP_OUT) / (1024 * 1024)
    print(f"\n✅ Created: {ZIP_OUT} ({sz_mb:.2f} MB)")


if __name__ == "__main__":
    main()
