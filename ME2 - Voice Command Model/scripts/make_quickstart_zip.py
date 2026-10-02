#!/usr/bin/env python3
"""
make_quickstart_zip.py

Creates ME2-quickstart.zip containing the minimal standalone files needed
to run demo_rpi5.py and simulate_demo.py on Raspberry Pi 5 without PyTorch.
"""

import os
import sys
import shutil
import zipfile

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DIST_DIR = os.path.join(BASE_DIR, "dist")
ZIP_NAME = "ME2-quickstart.zip"
ZIP_OUT = os.path.join(BASE_DIR, ZIP_NAME)

FILES_TO_PACK = [
    # Root scripts
    ("demo_rpi5.py", "demo_rpi5.py"),
    ("simulate_demo.py", "simulate_demo.py"),
    ("requirements-pi.txt", "requirements-pi.txt"),
    ("QUICKSTART.md", "QUICKSTART.md"),
    # Required data files
    ("data/labels_94.json", "data/labels_94.json"),
]

# Assets to place in exports/
EXPORTS_ASSETS = [
    ("exports/labels_94.json", "exports/labels_94.json"),
    ("exports/mel_filters_40.npy", "exports/mel_filters_40.npy"),
    ("exports/hann_window_400.npy", "exports/hann_window_400.npy"),
    ("exports/bcresnet_94class_int8.onnx", "exports/bcresnet_94class_int8.onnx"),
    ("exports/wakeword_int8.onnx", "exports/wakeword_int8.onnx"),
]


def resolve_asset_path(base_dir, src_rel):
    primary = os.path.join(base_dir, src_rel)
    if os.path.exists(primary):
        return primary
    # Fallback to v3_94class if looking in exports/
    if src_rel.startswith("exports/"):
        fname = os.path.basename(src_rel)
        v3_path = os.path.join(base_dir, "exports/v3_94class", fname)
        if os.path.exists(v3_path):
            return v3_path
    return primary


def main():
    print("=" * 65)
    print("📦 BUILDING STANDALONE ME2-quickstart.zip")
    print("=" * 65)

    if os.path.exists(ZIP_OUT):
        os.remove(ZIP_OUT)

    with zipfile.ZipFile(ZIP_OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        for src_rel, dst_rel in FILES_TO_PACK:
            src = resolve_asset_path(BASE_DIR, src_rel)
            if os.path.exists(src):
                zf.write(src, arcname=dst_rel)
                print(f"  + Added: {dst_rel}")
            else:
                print(f"  ! Warning: {src_rel} not found!")

        for src_rel, dst_rel in EXPORTS_ASSETS:
            src = resolve_asset_path(BASE_DIR, src_rel)
            if os.path.exists(src):
                zf.write(src, arcname=dst_rel)
                print(f"  + Added: {dst_rel} ({os.path.getsize(src) / 1024:.1f} KB)")
            else:
                print(f"  ! Warning: {src_rel} not found!")

    sz_mb = os.path.getsize(ZIP_OUT) / (1024 * 1024)
    print(f"\n✅ Created: {ZIP_OUT} ({sz_mb:.2f} MB)")


if __name__ == "__main__":
    main()
