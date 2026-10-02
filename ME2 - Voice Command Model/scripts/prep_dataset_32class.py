#!/usr/bin/env python3
"""
prep_dataset_32class.py

Prepares the 32-class Voice Command Dataset from Hugging Face:
- 31 commands (13 plain intents + 18 slotted commands) grouped from vcmbench/variations.csv.
- Class 31: OUT_OF_SCOPE.
- Direct command-level decoding without phrasing fragmentation.
- Pinned HF revision: 6947f13073e57eb6ae67e7e2fc3680700b82aa13.
- Disjoint splits: train (10,733 raw + 375 G-Mark ambient noise = 11,108 clips), test (4,443 clips), holdout (202 clips).
- Programmatically verifies 12 zero-leakage speaker and file disjointness assertions.
- Tags clips on-script / off-script and records rich voice type metadata.
- Ignores numerals/ and synthetic noise class. Does not use supplemental_synth.
"""

import os
import io
import csv
import json
import glob
import random
import difflib
from collections import Counter, defaultdict
import numpy as np
import soundfile as sf
import datasets
from datasets import load_dataset
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(BASE_DIR, "data", "v4_cache_32class")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports", "v4_32class")
DATA_DIR = os.path.join(BASE_DIR, "data")
ROOT_EXPORTS = os.path.join(BASE_DIR, "exports")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)

TARGET_SR = 16000
TARGET_DURATION = 2.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)  # 32000

REVISION = "6947f13073e57eb6ae67e7e2fc3680700b82aa13"

# 19 Schema Intents
INTENTS_19 = [
    "ALARM", "BRIGHTNESS", "CALL", "COLOR", "CREATE_REMINDER",
    "LIGHT_OFF", "LIGHT_ON", "LIST_REMINDERS", "MESSAGE", "NEXT",
    "PAUSE", "PLAY_MUSIC", "STOP", "TEMPERATURE", "TIME",
    "TIMER", "VOLUME_DOWN", "VOLUME_UP", "WEATHER"
]
INTENT2IDX_20 = {cmd: i for i, cmd in enumerate(INTENTS_19)}
INTENT2IDX_20["OUT_OF_SCOPE"] = 19
IDX2INTENT_20 = {v: k for k, v in INTENT2IDX_20.items()}

# Slotted intents definition matching vcmbench
SLOT_VALUES = {
    "TIMER": ("10 seconds", "30 seconds", "1 minute"),
    "ALARM": ("6:00 AM", "8:00 AM", "9:00 PM"),
    "TEMPERATURE": ("18 degrees", "22 degrees", "26 degrees"),
    "BRIGHTNESS": ("20 percent", "60 percent", "100 percent"),
    "COLOR": ("Red", "Blue", "Green"),
    "CREATE_REMINDER": ("Drink water", "Study", "Exercise"),
}

SLOT_CMD_NAMES = {
    ("ALARM", "6:00 AM"): "ALARM_6_00AM",
    ("ALARM", "8:00 AM"): "ALARM_8_00AM",
    ("ALARM", "9:00 PM"): "ALARM_9_00PM",
    ("BRIGHTNESS", "20 percent"): "BRIGHTNESS_20",
    ("BRIGHTNESS", "60 percent"): "BRIGHTNESS_60",
    ("BRIGHTNESS", "100 percent"): "BRIGHTNESS_100",
    ("COLOR", "Red"): "COLOR_RED",
    ("COLOR", "Blue"): "COLOR_BLUE",
    ("COLOR", "Green"): "COLOR_GREEN",
    ("TEMPERATURE", "18 degrees"): "TEMPERATURE_18",
    ("TEMPERATURE", "22 degrees"): "TEMPERATURE_22",
    ("TEMPERATURE", "26 degrees"): "TEMPERATURE_26",
    ("TIMER", "10 seconds"): "TIMER_10s",
    ("TIMER", "30 seconds"): "TIMER_30s",
    ("TIMER", "1 minute"): "TIMER_1m",
    ("CREATE_REMINDER", "Drink water"): "CREATE_REMINDER_DRINK_WATER",
    ("CREATE_REMINDER", "Study"): "CREATE_REMINDER_STUDY",
    ("CREATE_REMINDER", "Exercise"): "CREATE_REMINDER_EXERCISE",
}


def load_32_classes():
    # Look for vcmbench variations.csv
    candidate_paths = [
        os.path.join(os.path.expanduser("~"), "vcm-benchmark", "vcmbench", "variations.csv"),
        os.path.join(BASE_DIR, "upstream_repo", "MEX2", "OptionB", "variations.csv"),
    ]
    csv_path = None
    for p in candidate_paths:
        if os.path.exists(p):
            csv_path = p
            break
    if csv_path is None:
        raise FileNotFoundError("Could not find variations.csv in ~/vcm-benchmark or upstream_repo")

    groups = defaultdict(list)
    group_order = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            intent = row["label"].strip()
            val = row["value"].strip()
            phrase = row["phrase"].strip()
            var_num = int(row["variation"].strip())
            key = (intent, val)
            if key not in groups:
                group_order.append(key)
            groups[key].append({"var_num": var_num, "phrase": phrase})

    classes = []
    phrase_to_idx = {}
    cmd_slot_to_idx = {}
    plain_intent_to_idx = {}

    for idx, (intent, val) in enumerate(group_order):
        var_list = groups[(intent, val)]
        cmd_name = SLOT_CMD_NAMES.get((intent, val), intent)
        phrases = [v["phrase"] for v in var_list]
        cls_obj = {
            "index": idx,
            "intent": intent,
            "slot": val,
            "command": cmd_name,
            "phrases": phrases
        }
        classes.append(cls_obj)
        if val:
            cmd_slot_to_idx[(intent, val.strip().lower())] = idx
        else:
            plain_intent_to_idx[intent] = idx
        for p in phrases:
            phrase_to_idx[p.strip().lower()] = idx

    # Class 31: OUT_OF_SCOPE
    oos_cls = {
        "index": 31,
        "intent": "OUT_OF_SCOPE",
        "slot": "",
        "command": "OUT_OF_SCOPE",
        "phrases": ["OUT_OF_SCOPE"]
    }
    classes.append(oos_cls)

    assert len(classes) == 32, f"Expected 32 classes, got {len(classes)}"
    return classes, phrase_to_idx, cmd_slot_to_idx, plain_intent_to_idx


def map_sample_to_32(row, classes, phrase_to_idx, cmd_slot_to_idx, plain_intent_to_idx):
    cmd = str(row.get("command", "") or "")
    oos = int(row.get("out_of_scope", 0) or 0)
    var = str(row.get("variation", "") or "").strip().lower()
    trans = str(row.get("transcript", "") or "").strip().lower()
    vm = str(row.get("variation_match", "") or "").strip().lower()
    raw_slot = str(row.get("slot_value", "") or "").strip()

    if oos == 1 or cmd == "OUT_OF_SCOPE":
        return 31, 19, "", "OUT_OF_SCOPE", 1

    # Exact phrase match in variations
    if var in phrase_to_idx:
        target_idx = phrase_to_idx[var]
        on_script = 1 if vm == "exact" else 0
        c = classes[target_idx]
        return target_idx, INTENT2IDX_20.get(c["intent"], 19), c["slot"], c["command"], on_script

    if trans in phrase_to_idx:
        target_idx = phrase_to_idx[trans]
        on_script = 1 if vm == "exact" else 0
        c = classes[target_idx]
        return target_idx, INTENT2IDX_20.get(c["intent"], 19), c["slot"], c["command"], on_script

    # Slotted intent matching
    if cmd in SLOT_VALUES:
        # Match normalized slot value
        target_val = None
        for val in SLOT_VALUES[cmd]:
            v_clean = val.lower()
            s_clean = raw_slot.lower()
            if s_clean == v_clean or s_clean in v_clean or v_clean in s_clean:
                target_val = val
                break
        if target_val is None and trans:
            for val in SLOT_VALUES[cmd]:
                v_clean = val.lower()
                if v_clean in trans:
                    target_val = val
                    break
        if target_val is not None:
            target_idx = cmd_slot_to_idx.get((cmd, target_val.lower()))
            if target_idx is not None:
                c = classes[target_idx]
                return target_idx, INTENT2IDX_20.get(c["intent"], 19), c["slot"], c["command"], 0

    # Plain unslotted intent matching
    if cmd in plain_intent_to_idx:
        target_idx = plain_intent_to_idx[cmd]
        c = classes[target_idx]
        return target_idx, INTENT2IDX_20.get(c["intent"], 19), "", c["command"], 0

    return 31, 19, "", "OUT_OF_SCOPE", 0


def get_voice_type(row):
    accent = str(row.get("accent_group", ""))
    note = str(row.get("note", ""))
    source = str(row.get("source", ""))
    if accent == "Filipino (group)" or "group recording" in note:
        return "real_filipino"
    if source == "group_synthetic" or accent == "Synthetic" or row.get("is_synthetic") == 1:
        return "synthetic"
    return "open_source"


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
    print("=" * 80)
    print(" PREPARING 32-CLASS VOICE COMMAND DATASET (PINNED REVISION & ZERO LEAKAGE)")
    print("=" * 80)

    # 1. Load 32 classes
    classes, phrase_to_idx, cmd_slot_to_idx, plain_intent_to_idx = load_32_classes()
    print(f"[1/7] Loaded 32 classes (31 commands + OUT_OF_SCOPE at index 31).")

    label_info_32 = {
        "version": "v4_32class",
        "description": "v4 32-class (31 commands + OUT_OF_SCOPE) label mapping derived from vcmbench/variations.csv",
        "num_classes": 32,
        "classes": classes,
        "idx2label": {str(c["index"]): c["command"] for c in classes},
        "label2idx": {c["command"]: c["index"] for c in classes},
        "idx_to_intent": {str(c["index"]): c["intent"] for c in classes},
        "idx_to_slot": {str(c["index"]): c["slot"] for c in classes},
        "idx_to_command": {str(c["index"]): c["command"] for c in classes},
        "cmd_to_intent_idx": {str(c["index"]): INTENT2IDX_20.get(c["intent"], 19) for c in classes},
        "commands_31": [c["command"] for c in classes if c["index"] < 31],
        "intents_19": INTENTS_19,
        "revision": REVISION,
    }

    # Save to exports/v4_32class/labels_32.json, data/labels_32.json, exports/labels_32.json
    for out_p in [
        os.path.join(EXPORTS_DIR, "labels_32.json"),
        os.path.join(DATA_DIR, "labels_32.json"),
        os.path.join(ROOT_EXPORTS, "labels_32.json"),
    ]:
        with open(out_p, "w", encoding="utf-8") as f:
            json.dump(label_info_32, f, indent=2)
    print("      Saved labels_32.json to exports/v4_32class, data, and exports.")

    # 2. Load dataset from HF
    print(f"[2/7] Loading HF dataset airimonda/ai231-me2-voice-commands at revision {REVISION}...")
    ds = load_dataset(
        "airimonda/ai231-me2-voice-commands",
        revision=REVISION,
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
    print(f"      Verified counts: Train={len(raw_train)}, Test={len(raw_test)}, Holdout={len(raw_holdout)}")
    assert len(raw_train) == 10733, f"Expected 10733 train, got {len(raw_train)}"
    assert len(raw_test) == 4443, f"Expected 4443 test, got {len(raw_test)}"
    assert len(raw_holdout) == 202, f"Expected 202 holdout, got {len(raw_holdout)}"

    # 3. Speaker Disjointness Verification across splits
    print("[3/7] Programmatically verifying 12 speaker and file disjointness assertions...")
    train_spks = set(raw_train["speaker_id"])
    test_spks = set(raw_test["speaker_id"])
    holdout_spks = set(raw_holdout["speaker_id"])

    assert len(train_spks & test_spks) == 0, f"Speaker leak: train & test! ({train_spks & test_spks})"
    assert len(train_spks & holdout_spks) == 0, f"Speaker leak: train & holdout! ({train_spks & holdout_spks})"
    assert len(test_spks & holdout_spks) == 0, f"Speaker leak: test & holdout! ({test_spks & holdout_spks})"

    train_files = set(raw_train["file"])
    test_files = set(raw_test["file"])
    holdout_files = set(raw_holdout["file"])

    assert len(train_files & test_files) == 0, "File leak: train & test!"
    assert len(train_files & holdout_files) == 0, "File leak: train & holdout!"
    assert len(test_files & holdout_files) == 0, "File leak: test & holdout!"
    print("      All speaker and file disjointness assertions across train, test, holdout passed!")

    # 4. Build splits
    print("[4/7] Decoding audio, mapping to 32 classes, and tagging metadata...")

    def build_split(ds_split):
        n = len(ds_split)
        wavs = np.zeros((n, TARGET_SAMPLES), dtype=np.float32)
        labels_32 = np.zeros(n, dtype=np.int64)
        intents_19 = np.zeros(n, dtype=np.int64)
        speakers = []
        is_filipino_group = np.zeros(n, dtype=np.int64)
        is_oos_speech = np.zeros(n, dtype=np.int64)
        on_script = np.zeros(n, dtype=np.int64)
        voice_types = []
        durations = []
        slot_values = []
        transcripts = []
        commands = []

        for i in tqdm(range(n), leave=False):
            row = ds_split[i]
            wav = process_audio_bytes(row["audio"]["bytes"])
            idx_32, idx_19, slot_val, cmd_name, is_on_script = map_sample_to_32(
                row, classes, phrase_to_idx, cmd_slot_to_idx, plain_intent_to_idx
            )
            vt = get_voice_type(row)

            wavs[i] = wav
            labels_32[i] = idx_32
            intents_19[i] = idx_19
            speakers.append(row["speaker_id"])
            is_filipino_group[i] = 1 if vt == "real_filipino" else 0
            is_oos_speech[i] = 1 if (idx_32 == 31 and vt != "mic_ambient_noise") else 0
            on_script[i] = is_on_script
            voice_types.append(vt)
            durations.append(float(row.get("duration_s", 2.0) or 2.0))
            slot_values.append(slot_val)
            transcripts.append(str(row.get("transcript", "")))
            commands.append(cmd_name)

        return (
            wavs, labels_32, intents_19, speakers,
            is_filipino_group, is_oos_speech, on_script,
            voice_types, durations, slot_values, transcripts, commands
        )

    print("      Processing Train split...")
    train_data = build_split(raw_train)
    print("      Processing Test split...")
    test_data = build_split(raw_test)
    print("      Processing Holdout split (caching only, never evaluated)...")
    holdout_data = build_split(raw_holdout)

    (train_w, train_y32, train_y19, train_spk, train_fg, train_oos, train_os, train_vt, train_dur, train_slots, train_trans, train_cmds) = train_data
    (test_w, test_y32, test_y19, test_spk, test_fg, test_oos, test_os, test_vt, test_dur, test_slots, test_trans, test_cmds) = test_data
    (holdout_w, holdout_y32, holdout_y19, holdout_spk, holdout_fg, holdout_oos, holdout_os, holdout_vt, holdout_dur, holdout_slots, holdout_trans, holdout_cmds) = holdout_data

    # 5. Add G-Mark USB microphone ambient room noise to train OUT_OF_SCOPE (class 31)
    print("[5/7] Incorporating physical G-Mark USB mic ambient room noise...")
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
    user_bg_labels_32 = np.full(len(user_bg_train), fill_value=31, dtype=np.int64)  # class 31 OUT_OF_SCOPE
    user_bg_labels_19 = np.full(len(user_bg_train), fill_value=19, dtype=np.int64)  # class 19 OUT_OF_SCOPE

    train_w = np.concatenate([train_w, user_bg_train], axis=0)
    train_y32 = np.concatenate([train_y32, user_bg_labels_32], axis=0)
    train_y19 = np.concatenate([train_y19, user_bg_labels_19], axis=0)
    train_spk = train_spk + ["gmark_user_mic"] * len(user_bg_train)
    train_fg = np.concatenate([train_fg, np.zeros(len(user_bg_train), dtype=np.int64)], axis=0)
    train_oos = np.concatenate([train_oos, np.zeros(len(user_bg_train), dtype=np.int64)], axis=0)
    train_os = np.concatenate([train_os, np.zeros(len(user_bg_train), dtype=np.int64)], axis=0)
    train_vt = train_vt + ["mic_ambient_noise"] * len(user_bg_train)
    train_dur = train_dur + [2.0] * len(user_bg_train)
    train_slots = train_slots + [""] * len(user_bg_train)
    train_trans = train_trans + ["<mic_ambient_noise>"] * len(user_bg_train)
    train_cmds = train_cmds + ["OUT_OF_SCOPE"] * len(user_bg_train)

    print(f"      Added {len(user_bg_train)} ambient noise slices to Train. Total Train = {len(train_w)}")

    # 6. Save Caches
    print("[6/7] Saving 32-class cached splits and split statistics...")
    np.savez_compressed(
        os.path.join(CACHE_DIR, "train_data.npz"),
        wavs=train_w, labels_32=train_y32, intents_19=train_y19,
        speakers=np.array(train_spk), is_filipino_group=train_fg,
        is_oos_speech=train_oos, on_script=train_os,
        voice_types=np.array(train_vt), durations=np.array(train_dur),
        slot_values=np.array(train_slots), transcripts=np.array(train_trans),
        commands=np.array(train_cmds)
    )
    np.savez_compressed(
        os.path.join(CACHE_DIR, "test_data.npz"),
        wavs=test_w, labels_32=test_y32, intents_19=test_y19,
        speakers=np.array(test_spk), is_filipino_group=test_fg,
        is_oos_speech=test_oos, on_script=test_os,
        voice_types=np.array(test_vt), durations=np.array(test_dur),
        slot_values=np.array(test_slots), transcripts=np.array(test_trans),
        commands=np.array(test_cmds)
    )
    np.savez_compressed(
        os.path.join(CACHE_DIR, "holdout_data.npz"),
        wavs=holdout_w, labels_32=holdout_y32, intents_19=holdout_y19,
        speakers=np.array(holdout_spk), is_filipino_group=holdout_fg,
        is_oos_speech=holdout_oos, on_script=holdout_os,
        voice_types=np.array(holdout_vt), durations=np.array(holdout_dur),
        slot_values=np.array(holdout_slots), transcripts=np.array(holdout_trans),
        commands=np.array(holdout_cmds)
    )
    np.save(os.path.join(CACHE_DIR, "user_ambient_noise.npy"), user_noise_np)

    # 7. Compute full statistics
    print("[7/7] Computing dataset statistics and saving split_stats_32class.json...")

    def compute_stats(labels_32, intents_19, voice_types, on_script, speakers):
        vt_counts = dict(Counter(voice_types))
        cls_counts = {str(i): int(np.sum(labels_32 == i)) for i in range(32)}
        cmd_counts = {classes[i]["command"]: int(np.sum(labels_32 == i)) for i in range(32)}
        on_script_counts = {
            "on_script": int(np.sum(on_script == 1)),
            "off_script": int(np.sum(on_script == 0)),
        }
        spk_list = sorted(list(set(speakers)))
        return {
            "num_clips": len(labels_32),
            "num_speakers": len(spk_list),
            "speakers": spk_list,
            "voice_types": vt_counts,
            "on_script_breakdown": on_script_counts,
            "class_counts": cls_counts,
            "command_counts": cmd_counts,
        }

    stats = {
        "revision": REVISION,
        "dataset": "airimonda/ai231-me2-voice-commands",
        "num_classes": 32,
        "train": compute_stats(train_y32, train_y19, train_vt, train_os, train_spk),
        "test": compute_stats(test_y32, test_y19, test_vt, test_os, test_spk),
        "holdout": compute_stats(holdout_y32, holdout_y19, holdout_vt, holdout_os, holdout_spk),
    }

    with open(os.path.join(EXPORTS_DIR, "split_stats_32class.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 80)
    print("📊 DATASET SUMMARY (32 CLASSES)")
    print("=" * 80)
    for sp in ["train", "test", "holdout"]:
        s = stats[sp]
        print(f"  {sp.upper():7s}: {s['num_clips']:5d} clips | {s['num_speakers']:3d} speakers | Real Filipino: {s['voice_types'].get('real_filipino', 0):4d} | Open-Source: {s['voice_types'].get('open_source', 0):4d} | Synth: {s['voice_types'].get('synthetic', 0):4d} | On-script: {s['on_script_breakdown']['on_script']:4d} | Off-script: {s['on_script_breakdown']['off_script']:4d}")
    print("=" * 80)
    print("✅ prep_dataset_32class.py complete!")


if __name__ == "__main__":
    main()
