#!/usr/bin/env python3
"""
prep_dataset_20class.py

Prepares the 20-class Voice Command Dataset from Hugging Face:
- 19 schema commands: ALARM, BRIGHTNESS, CALL, COLOR, CREATE_REMINDER, LIGHT_OFF,
  LIGHT_ON, LIST_REMINDERS, MESSAGE, NEXT, PAUSE, PLAY_MUSIC, STOP, TEMPERATURE,
  TIME, TIMER, VOLUME_DOWN, VOLUME_UP, WEATHER.
- 20th class: OUT_OF_SCOPE (index 19 in 0-based indexing).
- Preserves speaker-disjoint validation carve-out from train.
- Preserves exact test and holdout splits (with held-out flag).
- Incorporates physical G-Mark USB microphone ambient room noise into OUT_OF_SCOPE.
- Records rich metadata for evaluation: is_filipino_group, is_oos_speech, accent_group.
- Executes 12 programmatic zero-leak assertions.
"""

import os
import io
import json
import glob
import random
from collections import Counter, defaultdict
import numpy as np
import soundfile as sf
import datasets
from datasets import load_dataset
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "v2_cache_20class")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports", "v2_20class")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)

TARGET_SR = 16000
TARGET_DURATION = 2.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)  # 32000

# 20-Class Schema: 19 commands + OUT_OF_SCOPE as 20th class (index 19)
COMMANDS_19 = [
    "ALARM", "BRIGHTNESS", "CALL", "COLOR", "CREATE_REMINDER",
    "LIGHT_OFF", "LIGHT_ON", "LIST_REMINDERS", "MESSAGE", "NEXT",
    "PAUSE", "PLAY_MUSIC", "STOP", "TEMPERATURE", "TIME",
    "TIMER", "VOLUME_DOWN", "VOLUME_UP", "WEATHER"
]

LABEL2IDX_20 = {cmd: i for i, cmd in enumerate(COMMANDS_19)}
LABEL2IDX_20["OUT_OF_SCOPE"] = 19  # 20th class (index 19)
IDX2LABEL_20 = {v: k for k, v in LABEL2IDX_20.items()}


def map_row_to_class(cmd, oos):
    if oos == 1 or cmd == "OUT_OF_SCOPE":
        return "OUT_OF_SCOPE", 19
    if cmd in LABEL2IDX_20:
        return cmd, LABEL2IDX_20[cmd]
    return "OUT_OF_SCOPE", 19


def process_audio_bytes(audio_bytes):
    with io.BytesIO(audio_bytes) as bio:
        data, sr = sf.read(bio, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != TARGET_SR:
        num_s = int(len(data) * TARGET_SR / sr)
        orig_idx = np.arange(len(data))
        target_idx = np.linspace(0, len(data) - 1, num_s)
        data = np.interp(target_idx, orig_idx, data).astype(np.float32)
    if len(data) < TARGET_SAMPLES:
        pad = TARGET_SAMPLES - len(data)
        data = np.pad(data, (pad // 2, pad - pad // 2), mode="constant")
    elif len(data) > TARGET_SAMPLES:
        start = (len(data) - TARGET_SAMPLES) // 2
        data = data[start : start + TARGET_SAMPLES]
    return data.astype(np.float32)


def main():
    print("=" * 70)
    print("📥 PREPARING 20-CLASS VOICE COMMAND DATASET (19 COMMANDS + OUT_OF_SCOPE)")
    print("=" * 70)

    # 1. Load Dataset
    print("[1/6] Loading dataset from airimonda/ai231-me2-voice-commands...")
    ds = load_dataset(
        "airimonda/ai231-me2-voice-commands",
        data_files={
            "train": "data/train-*",
            "test": "data/test-*",
            "holdout": "data/holdout-*",
        },
        verification_mode="no_checks",
    )
    ds = ds.cast_column("audio", datasets.Audio(decode=False))

    raw_train = ds["train"]
    raw_test = ds["test"]
    raw_holdout = ds["holdout"]
    print(f"      Clips downloaded: Train={len(raw_train)}, Test={len(raw_test)}, Holdout={len(raw_holdout)}")

    # 2. Carve out speaker-disjoint Validation set from Train
    print("[2/6] Carving out speaker-disjoint validation set from train...")
    spk_to_train_indices = defaultdict(list)
    for idx, spk in enumerate(raw_train["speaker_id"]):
        spk_to_train_indices[spk].append(idx)

    train_speakers_pool = sorted(spk_to_train_indices.keys())
    rng = random.Random(42)
    val_spk_set = set(rng.sample(train_speakers_pool, k=38))  # ~12% of speakers
    train_spk_set = set(train_speakers_pool) - val_spk_set

    val_indices = [idx for spk in sorted(val_spk_set) for idx in spk_to_train_indices[spk]]
    train_indices = [idx for spk in sorted(train_spk_set) for idx in spk_to_train_indices[spk]]

    test_spk_set = set(raw_test["speaker_id"])
    holdout_spk_set = set(raw_holdout["speaker_id"])

    # 3. Assertions
    print("[3/6] Programmatically asserting zero speaker and file leakage...")
    assert len(train_spk_set & val_spk_set) == 0, "Speaker leak: train & val!"
    assert len(train_spk_set & test_spk_set) == 0, "Speaker leak: train & test!"
    assert len(train_spk_set & holdout_spk_set) == 0, "Speaker leak: train & holdout!"
    assert len(val_spk_set & test_spk_set) == 0, "Speaker leak: val & test!"
    assert len(val_spk_set & holdout_spk_set) == 0, "Speaker leak: val & holdout!"
    assert len(test_spk_set & holdout_spk_set) == 0, "Speaker leak: test & holdout!"

    train_files = set(raw_train["file"][i] for i in train_indices)
    val_files = set(raw_train["file"][i] for i in val_indices)
    test_files = set(raw_test["file"])
    holdout_files = set(raw_holdout["file"])

    assert len(train_files & val_files) == 0, "File leak: train & val!"
    assert len(train_files & test_files) == 0, "File leak: train & test!"
    assert len(train_files & holdout_files) == 0, "File leak: train & holdout!"
    assert len(val_files & test_files) == 0, "File leak: val & test!"
    assert len(val_files & holdout_files) == 0, "File leak: val & holdout!"
    assert len(test_files & holdout_files) == 0, "File leak: test & holdout!"
    print("      ✅ All 12 speaker and file disjointness assertions passed!")

    # 4. Process Audio & Metadata
    print("[4/6] Decoding waveforms and tagging evaluation subsets...")

    def build_split(ds_split, indices=None):
        indices = list(range(len(ds_split))) if indices is None else indices
        wavs = np.zeros((len(indices), TARGET_SAMPLES), dtype=np.float32)
        labels = np.zeros(len(indices), dtype=np.int64)
        speakers = []
        is_filipino_group = np.zeros(len(indices), dtype=np.int64)
        is_oos_speech = np.zeros(len(indices), dtype=np.int64)
        durations = []

        for out_idx, in_idx in enumerate(tqdm(indices, leave=False)):
            row = ds_split[in_idx]
            wav = process_audio_bytes(row["audio"]["bytes"])
            lbl_name, lbl_idx = map_row_to_class(row["command"], row["out_of_scope"])
            wavs[out_idx] = wav
            labels[out_idx] = lbl_idx
            speakers.append(row["speaker_id"])
            durations.append(row["duration_s"] if row["duration_s"] is not None else 2.0)

            # Tag Filipino group in-scope recordings
            accent = str(row.get("accent_group", ""))
            note = str(row.get("note", ""))
            if (accent == "Filipino (group)" or "group recording" in note) and lbl_idx != 19:
                is_filipino_group[out_idx] = 1

            # Tag Out-of-scope speech
            if lbl_idx == 19:
                is_oos_speech[out_idx] = 1

        return wavs, labels, speakers, is_filipino_group, is_oos_speech, durations

    train_w, train_y, train_spk, train_fg, train_oos, train_dur = build_split(raw_train, train_indices)
    val_w, val_y, val_spk, val_fg, val_oos, val_dur = build_split(raw_train, val_indices)
    test_w, test_y, test_spk, test_fg, test_oos, test_dur = build_split(raw_test)
    holdout_w, holdout_y, holdout_spk, holdout_fg, holdout_oos, holdout_dur = build_split(raw_holdout)

    # 5. Add G-Mark USB Ambient Room Noise to Train (OUT_OF_SCOPE class 19)
    print("[5/6] Incorporating physical G-Mark USB mic ambient room noise...")
    noise_paths = sorted(glob.glob(os.path.join(BASE_DIR, "data", "my_wakeword", "idle_noise_*.wav")))
    user_noise_clips = []
    for np_f in noise_paths:
        d, sr = sf.read(np_f, dtype="float32")
        if d.ndim > 1:
            d = np.mean(d, axis=1)
        if sr != TARGET_SR:
            num_s = int(len(d) * TARGET_SR / sr)
            orig_idx = np.arange(len(d))
            target_idx = np.linspace(0, len(d) - 1, num_s)
            d = np.interp(target_idx, orig_idx, d).astype(np.float32)
        if len(d) < TARGET_SAMPLES:
            repeats = int(np.ceil(TARGET_SAMPLES / len(d)))
            d = np.tile(d, repeats)[:TARGET_SAMPLES]
        else:
            d = d[:TARGET_SAMPLES]
        user_noise_clips.append(d)

    user_noise_np = np.array(user_noise_clips, dtype=np.float32)

    user_bg_train = []
    for n in user_noise_np:
        for _ in range(25):
            user_bg_train.append(n)
    user_bg_train = np.array(user_bg_train, dtype=np.float32)
    user_bg_labels = np.full(len(user_bg_train), fill_value=19, dtype=np.int64)  # class 19 OUT_OF_SCOPE

    train_w = np.concatenate([train_w, user_bg_train], axis=0)
    train_y = np.concatenate([train_y, user_bg_labels], axis=0)
    train_spk = train_spk + ["gmark_user_mic"] * len(user_bg_train)
    train_fg = np.concatenate([train_fg, np.zeros(len(user_bg_train), dtype=np.int64)], axis=0)
    train_oos = np.concatenate([train_oos, np.zeros(len(user_bg_train), dtype=np.int64)], axis=0)
    train_dur = train_dur + [2.0] * len(user_bg_train)

    # 6. Save Caches & Label JSON
    print("[6/6] Saving 20-class cached splits and metadata...")
    np.savez_compressed(
        os.path.join(CACHE_DIR, "train_data.npz"),
        wavs=train_w, labels=train_y, speakers=np.array(train_spk),
        is_filipino_group=train_fg, is_oos_speech=train_oos
    )
    np.savez_compressed(
        os.path.join(CACHE_DIR, "val_data.npz"),
        wavs=val_w, labels=val_y, speakers=np.array(val_spk),
        is_filipino_group=val_fg, is_oos_speech=val_oos
    )
    np.savez_compressed(
        os.path.join(CACHE_DIR, "test_data.npz"),
        wavs=test_w, labels=test_y, speakers=np.array(test_spk),
        is_filipino_group=test_fg, is_oos_speech=test_oos
    )
    np.savez_compressed(
        os.path.join(CACHE_DIR, "holdout_data.npz"),
        wavs=holdout_w, labels=holdout_y, speakers=np.array(holdout_spk),
        is_filipino_group=holdout_fg, is_oos_speech=holdout_oos
    )
    np.save(os.path.join(CACHE_DIR, "user_ambient_noise.npy"), user_noise_np)

    # Save Label Map
    label_info_20 = {
        "num_classes": 20,
        "commands_19": COMMANDS_19,
        "label2idx": LABEL2IDX_20,
        "idx2label": {str(k): v for k, v in IDX2LABEL_20.items()},
        "oos_class_index": 19,
    }
    with open(os.path.join(EXPORTS_DIR, "labels_20.json"), "w") as f:
        json.dump(label_info_20, f, indent=2)

    # Compute and display statistics
    stats = {
        "train": {
            "utterances": len(train_y),
            "hours": round(float(sum(train_dur) / 3600.0), 3),
            "speakers": len(set(train_spk)),
            "oos_count": int(np.sum(train_y == 19)),
            "class_distribution": {IDX2LABEL_20[int(k)]: int(v) for k, v in sorted(Counter(train_y).items())},
        },
        "val": {
            "utterances": len(val_y),
            "hours": round(float(sum(val_dur) / 3600.0), 3),
            "speakers": len(set(val_spk)),
            "oos_count": int(np.sum(val_y == 19)),
            "filipino_in_scope_count": int(np.sum(val_fg)),
            "class_distribution": {IDX2LABEL_20[int(k)]: int(v) for k, v in sorted(Counter(val_y).items())},
        },
        "test": {
            "utterances": len(test_y),
            "hours": round(float(sum(test_dur) / 3600.0), 3),
            "speakers": len(set(test_spk)),
            "oos_count": int(np.sum(test_y == 19)),
            "filipino_in_scope_count": int(np.sum(test_fg)),
            "class_distribution": {IDX2LABEL_20[int(k)]: int(v) for k, v in sorted(Counter(test_y).items())},
        },
        "holdout": {
            "utterances": len(holdout_y),
            "hours": round(float(sum(holdout_dur) / 3600.0), 3),
            "speakers": len(set(holdout_spk)),
            "oos_count": int(np.sum(holdout_y == 19)),
            "filipino_in_scope_count": int(np.sum(holdout_fg)),
            "class_distribution": {IDX2LABEL_20[int(k)]: int(v) for k, v in sorted(Counter(holdout_y).items())},
        },
    }

    with open(os.path.join(EXPORTS_DIR, "split_stats_20class.json"), "w") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 70)
    print("📊 20-CLASS DATASET SUMMARY:")
    print("=" * 70)
    for sp_name, s in stats.items():
        oos_str = f"OOS: {s['oos_count']:3d}"
        fg_str = f" | Filipino Group: {s.get('filipino_in_scope_count', 0):3d}" if "filipino_in_scope_count" in s else ""
        print(f"  {sp_name.upper():<7} | Utterances: {s['utterances']:5d} | Hours: {s['hours']:5.2f} h | Speakers: {s['speakers']:3d} | {oos_str}{fg_str}")

    print(f"\nSaved metadata to: {os.path.join(EXPORTS_DIR, 'labels_20.json')}")
    print("✅ 20-Class Dataset Preparation Complete!")


if __name__ == "__main__":
    main()
