#!/usr/bin/env python3
"""
prep_dataset_94class.py

Prepares the 94-class Voice Command Dataset from Hugging Face:
- 93 command variations from vcmbench/variations.csv + OUT_OF_SCOPE (index 93).
- Direct decoding: variation index maps directly to (intent, slot_value, phrase).
- Solves Filipino speech starvation:
    - Retains 608 of 680 real Filipino clips in TRAIN (speakers: 202322013, 202322013_speaker2, S1, S2, S3).
    - Reserves exactly 1 unseen Filipino speaker in VAL (202521746 with 72 clips).
    - Never splits any speaker across train and val.
    - Leaves TEST (189 Filipino clips, 3 speakers) and HOLDOUT (73 Filipino clips, 1 speaker) untouched.
- Incorporates physical G-Mark USB microphone ambient room noise into OUT_OF_SCOPE in train.
- Records rich metadata for evaluation:
    - is_filipino_group (1 for real Filipino speech, 0 otherwise)
    - is_oos_speech (1 for speech OOS, 0 for noise/in-scope)
    - voice_type ('real_filipino', 'open_source', 'synthetic', 'mic_ambient_noise')
    - on_script (1 for exact scripted variation, 0 for natural paraphrase/off-script)
- Executes 12 programmatic zero-leak assertions.
- Also prepares supplemental_synth (train split only) for data scale ablation.
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
CACHE_DIR = os.path.join(BASE_DIR, "data", "v3_cache_94class")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports", "v3_94class")
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(EXPORTS_DIR, exist_ok=True)

TARGET_SR = 16000
TARGET_DURATION = 2.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)  # 32000

REVISION = "6947f13073e57eb6ae67e7e2fc3680700b82aa13"
BENCHMARK_VARIATIONS_CSV = "/home/misael.andre.maningo/vcm-benchmark/vcmbench/variations.csv"

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


def load_variations():
    variations = []
    phrase_to_idx = {}
    cmd_to_vars = defaultdict(list)
    with open(BENCHMARK_VARIATIONS_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for idx, row in enumerate(reader):
            v = {
                "index": idx,
                "intent": row["label"].strip(),
                "variation": row["variation"].strip(),
                "slot": row["value"].strip(),
                "phrase": row["phrase"].strip(),
            }
            variations.append(v)
            phrase_to_idx[v["phrase"].lower()] = idx
            cmd_to_vars[v["intent"]].append(v)

    # Class 93 is OUT_OF_SCOPE
    oos_class = {
        "index": 93,
        "intent": "OUT_OF_SCOPE",
        "variation": "",
        "slot": "",
        "phrase": "OUT_OF_SCOPE",
    }
    variations.append(oos_class)
    return variations, phrase_to_idx, cmd_to_vars


def map_sample_to_94(row, phrase_to_idx, cmd_to_vars):
    cmd = str(row.get("command", "") or "")
    oos = int(row.get("out_of_scope", 0) or 0)
    var = str(row.get("variation", "") or "").strip().lower()
    trans = str(row.get("transcript", "") or "").strip().lower()
    vm = str(row.get("variation_match", "") or "").strip().lower()
    note = str(row.get("note", "") or "").strip().lower()

    if oos == 1 or cmd == "OUT_OF_SCOPE":
        # Out of scope
        return 93, 19, "", "OUT_OF_SCOPE", 1

    # Exact phrase match in variations
    if var in phrase_to_idx:
        idx = phrase_to_idx[var]
        on_script = 1 if (vm in ("exact", "true", "1") or "scripted variation" in note) else 1
        return idx, INTENT2IDX_20.get(cmd, 19), row.get("slot_value", "") or "", var, on_script

    if trans in phrase_to_idx:
        idx = phrase_to_idx[trans]
        return idx, INTENT2IDX_20.get(cmd, 19), row.get("slot_value", "") or "", trans, 1

    # Fallback fuzzy match against candidate variations for this intent
    candidates = cmd_to_vars.get(cmd, [])
    if candidates:
        best = max(candidates, key=lambda c: difflib.SequenceMatcher(None, trans, c["phrase"].lower()).ratio())
        return best["index"], INTENT2IDX_20.get(cmd, 19), best["slot"], best["phrase"], 0

    return 93, 19, "", "OUT_OF_SCOPE", 0


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
    print("📥 PREPARING 94-CLASS VOICE COMMAND DATASET (PINNED REVISION & ZERO LEAKAGE)")
    print("=" * 80)

    # 1. Load variations
    variations, phrase_to_idx, cmd_to_vars = load_variations()
    assert len(variations) == 94, f"Expected 94 variations, got {len(variations)}"
    print(f"[1/7] Loaded 94 classes from {BENCHMARK_VARIATIONS_CSV} (93 variations + OUT_OF_SCOPE).")

    # Save labels_94.json
    label_info_94 = {
        "num_classes": 94,
        "classes": variations,
        "idx_to_intent": {str(v["index"]): v["intent"] for v in variations},
        "idx_to_slot": {str(v["index"]): v["slot"] for v in variations},
        "idx_to_phrase": {str(v["index"]): v["phrase"] for v in variations},
        "var_to_intent_idx": {str(v["index"]): INTENT2IDX_20.get(v["intent"], 19) for v in variations},
        "commands_19": INTENTS_19,
        "revision": REVISION,
    }
    with open(os.path.join(EXPORTS_DIR, "labels_94.json"), "w", encoding="utf-8") as f:
        json.dump(label_info_94, f, indent=2)
    with open(os.path.join(DATA_DIR, "labels_94.json"), "w", encoding="utf-8") as f:
        json.dump(label_info_94, f, indent=2)
    print("      Saved labels_94.json to exports and data dirs.")

    # 2. Load dataset from HF at pinned revision
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

    # 3. Speaker-disjoint Validation Set Carve-Out
    print("[3/7] Partitioning train and validation by speaker...")
    spk_to_train_indices = defaultdict(list)
    spk_is_filipino = {}
    spk_has_oos = defaultdict(int)

    for idx, row in enumerate(raw_train):
        spk = row["speaker_id"]
        spk_to_train_indices[spk].append(idx)
        vt = get_voice_type(row)
        if vt == "real_filipino":
            spk_is_filipino[spk] = True
        elif spk not in spk_is_filipino:
            spk_is_filipino[spk] = False

        if row["command"] == "OUT_OF_SCOPE" or row.get("out_of_scope") == 1:
            spk_has_oos[spk] += 1

    filipino_train_spks = sorted([s for s, is_f in spk_is_filipino.items() if is_f])
    non_filipino_train_spks = sorted([s for s, is_f in spk_is_filipino.items() if not is_f])
    non_fil_oos_spks = sorted([s for s in non_filipino_train_spks if spk_has_oos[s] > 0])
    non_fil_no_oos_spks = sorted([s for s in non_filipino_train_spks if spk_has_oos[s] == 0])

    print(f"      Filipino speakers in raw train ({len(filipino_train_spks)}): {filipino_train_spks}")
    print(f"      Non-Filipino speakers in raw train: {len(non_filipino_train_spks)}")

    # Reserve 202521746 (72 clips) for validation; the remaining 5 speakers (608 clips) remain in train!
    val_fil_spk = {"202521746"}
    train_fil_spk = set(filipino_train_spks) - val_fil_spk

    rng = random.Random(42)
    val_non_fil = set(rng.sample(non_fil_oos_spks, k=20)) | set(rng.sample(non_fil_no_oos_spks, k=15))
    train_non_fil = set(non_filipino_train_spks) - val_non_fil

    val_spk_set = val_fil_spk | val_non_fil
    train_spk_set = train_fil_spk | train_non_fil

    val_indices = [idx for spk in sorted(val_spk_set) for idx in spk_to_train_indices[spk]]
    train_indices = [idx for spk in sorted(train_spk_set) for idx in spk_to_train_indices[spk]]

    test_spk_set = set(raw_test["speaker_id"])
    holdout_spk_set = set(raw_holdout["speaker_id"])

    # Zero leakage programmatic assertions
    print("[4/7] Programmatically verifying 12 speaker and file disjointness assertions...")
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

    # 4. Build splits
    print("[5/7] Decoding audio, mapping to 94 classes, and tagging metadata...")

    def build_split(ds_split, indices=None):
        indices = list(range(len(ds_split))) if indices is None else indices
        wavs = np.zeros((len(indices), TARGET_SAMPLES), dtype=np.float32)
        labels_94 = np.zeros(len(indices), dtype=np.int64)
        intents_19 = np.zeros(len(indices), dtype=np.int64)
        speakers = []
        is_filipino_group = np.zeros(len(indices), dtype=np.int64)
        is_oos_speech = np.zeros(len(indices), dtype=np.int64)
        on_script = np.zeros(len(indices), dtype=np.int64)
        voice_types = []
        durations = []
        slot_values = []
        transcripts = []

        for out_idx, in_idx in enumerate(tqdm(indices, leave=False)):
            row = ds_split[in_idx]
            wav = process_audio_bytes(row["audio"]["bytes"])
            idx_94, idx_19, slot_val, phrase_txt, is_on_script = map_sample_to_94(row, phrase_to_idx, cmd_to_vars)
            vt = get_voice_type(row)

            wavs[out_idx] = wav
            labels_94[out_idx] = idx_94
            intents_19[out_idx] = idx_19
            speakers.append(row["speaker_id"])
            is_filipino_group[out_idx] = 1 if vt == "real_filipino" else 0
            is_oos_speech[out_idx] = 1 if (idx_94 == 93 and vt != "mic_ambient_noise") else 0
            on_script[out_idx] = is_on_script
            voice_types.append(vt)
            durations.append(float(row.get("duration_s", 2.0) or 2.0))
            slot_values.append(slot_val)
            transcripts.append(str(row.get("transcript", "")))

        return (
            wavs, labels_94, intents_19, speakers,
            is_filipino_group, is_oos_speech, on_script,
            voice_types, durations, slot_values, transcripts
        )

    train_data = build_split(raw_train, train_indices)
    val_data = build_split(raw_train, val_indices)
    test_data = build_split(raw_test)
    holdout_data = build_split(raw_holdout)

    (train_w, train_y94, train_y19, train_spk, train_fg, train_oos, train_os, train_vt, train_dur, train_slots, train_trans) = train_data
    (val_w, val_y94, val_y19, val_spk, val_fg, val_oos, val_os, val_vt, val_dur, val_slots, val_trans) = val_data
    (test_w, test_y94, test_y19, test_spk, test_fg, test_oos, test_os, test_vt, test_dur, test_slots, test_trans) = test_data
    (holdout_w, holdout_y94, holdout_y19, holdout_spk, holdout_fg, holdout_oos, holdout_os, holdout_vt, holdout_dur, holdout_slots, holdout_trans) = holdout_data

    # 5. Add G-Mark USB microphone ambient room noise to train (OUT_OF_SCOPE class 93)
    print("[6/7] Incorporating physical G-Mark USB mic ambient room noise...")
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
    user_bg_labels_94 = np.full(len(user_bg_train), fill_value=93, dtype=np.int64)  # class 93 OUT_OF_SCOPE
    user_bg_labels_19 = np.full(len(user_bg_train), fill_value=19, dtype=np.int64)  # class 19 OUT_OF_SCOPE

    train_w = np.concatenate([train_w, user_bg_train], axis=0)
    train_y94 = np.concatenate([train_y94, user_bg_labels_94], axis=0)
    train_y19 = np.concatenate([train_y19, user_bg_labels_19], axis=0)
    train_spk = train_spk + ["gmark_user_mic"] * len(user_bg_train)
    train_fg = np.concatenate([train_fg, np.zeros(len(user_bg_train), dtype=np.int64)], axis=0)
    train_oos = np.concatenate([train_oos, np.zeros(len(user_bg_train), dtype=np.int64)], axis=0)
    train_os = np.concatenate([train_os, np.zeros(len(user_bg_train), dtype=np.int64)], axis=0)
    train_vt = train_vt + ["mic_ambient_noise"] * len(user_bg_train)
    train_dur = train_dur + [2.0] * len(user_bg_train)
    train_slots = train_slots + [""] * len(user_bg_train)
    train_trans = train_trans + ["<mic_ambient_noise>"] * len(user_bg_train)

    # 6. Save caches
    print("[7/7] Saving 94-class cached splits and split statistics...")
    np.savez_compressed(
        os.path.join(CACHE_DIR, "train_data.npz"),
        wavs=train_w, labels_94=train_y94, intents_19=train_y19,
        speakers=np.array(train_spk), is_filipino_group=train_fg,
        is_oos_speech=train_oos, on_script=train_os,
        voice_types=np.array(train_vt), durations=np.array(train_dur),
        slot_values=np.array(train_slots), transcripts=np.array(train_trans)
    )
    np.savez_compressed(
        os.path.join(CACHE_DIR, "val_data.npz"),
        wavs=val_w, labels_94=val_y94, intents_19=val_y19,
        speakers=np.array(val_spk), is_filipino_group=val_fg,
        is_oos_speech=val_oos, on_script=val_os,
        voice_types=np.array(val_vt), durations=np.array(val_dur),
        slot_values=np.array(val_slots), transcripts=np.array(val_trans)
    )
    np.savez_compressed(
        os.path.join(CACHE_DIR, "test_data.npz"),
        wavs=test_w, labels_94=test_y94, intents_19=test_y19,
        speakers=np.array(test_spk), is_filipino_group=test_fg,
        is_oos_speech=test_oos, on_script=test_os,
        voice_types=np.array(test_vt), durations=np.array(test_dur),
        slot_values=np.array(test_slots), transcripts=np.array(test_trans)
    )
    np.savez_compressed(
        os.path.join(CACHE_DIR, "holdout_data.npz"),
        wavs=holdout_w, labels_94=holdout_y94, intents_19=holdout_y19,
        speakers=np.array(holdout_spk), is_filipino_group=holdout_fg,
        is_oos_speech=holdout_oos, on_script=holdout_os,
        voice_types=np.array(holdout_vt), durations=np.array(holdout_dur),
        slot_values=np.array(holdout_slots), transcripts=np.array(holdout_trans)
    )
    np.save(os.path.join(CACHE_DIR, "user_ambient_noise.npy"), user_noise_np)

    # 7. Supplemental synth train cache for ablation
    print("      Preparing supplemental_synth (train split only, 0 leakage)...")
    ds_supp = load_dataset(
        "airimonda/ai231-me2-voice-commands",
        "supplemental_synth",
        revision=REVISION,
        split="train",
        verification_mode="no_checks",
    )
    ds_supp = ds_supp.cast_column("audio", datasets.Audio(decode=False))
    supp_train_indices = [i for i, r in enumerate(ds_supp) if r["voice_split"] == "train"]
    print(f"      Supplemental synth train clips: {len(supp_train_indices)}")

    supp_w, supp_y94, supp_y19, supp_spk, supp_fg, supp_oos, supp_os, supp_vt, supp_dur, supp_slots, supp_trans = build_split(ds_supp, supp_train_indices)
    np.savez_compressed(
        os.path.join(CACHE_DIR, "synth_supp_train.npz"),
        wavs=supp_w, labels_94=supp_y94, intents_19=supp_y19,
        speakers=np.array(supp_spk), is_filipino_group=supp_fg,
        is_oos_speech=supp_oos, on_script=supp_os,
        voice_types=np.array(supp_vt), durations=np.array(supp_dur),
        slot_values=np.array(supp_slots), transcripts=np.array(supp_trans)
    )

    # Compute Split Stats
    def get_stats(wavs, labels_94, speakers, fg, oos, os_arr, vt, dur):
        total_hours = float(np.sum(dur) / 3600.0)
        c_94 = Counter(int(x) for x in labels_94)
        c_vt = Counter(vt)
        spk_set = sorted(list(set(speakers)))
        return {
            "num_clips": len(wavs),
            "num_speakers": len(spk_set),
            "speakers": spk_set,
            "total_hours": round(total_hours, 2),
            "filipino_speech_clips": int(np.sum(fg)),
            "oos_speech_clips": int(np.sum(oos)),
            "on_script_clips": int(np.sum(os_arr)),
            "off_script_clips": int(len(wavs) - np.sum(os_arr)),
            "voice_types": dict(c_vt),
            "class_distribution": {k: int(v) for k, v in c_94.items()},
        }

    stats = {
        "revision": REVISION,
        "dataset": "airimonda/ai231-me2-voice-commands",
        "num_classes": 94,
        "train": get_stats(train_w, train_y94, train_spk, train_fg, train_oos, train_os, train_vt, train_dur),
        "val": get_stats(val_w, val_y94, val_spk, val_fg, val_oos, val_os, val_vt, val_dur),
        "test": get_stats(test_w, test_y94, test_spk, test_fg, test_oos, test_os, test_vt, test_dur),
        "holdout": get_stats(holdout_w, holdout_y94, holdout_spk, holdout_fg, holdout_oos, holdout_os, holdout_vt, holdout_dur),
    }

    with open(os.path.join(CACHE_DIR, "split_stats_94class.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    with open(os.path.join(EXPORTS_DIR, "split_stats_94class.json"), "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 80)
    print("✅ 94-CLASS DATASET PREPARATION COMPLETE!")
    print(f"   Train: {len(train_w)} clips, {stats['train']['total_hours']}h ({stats['train']['filipino_speech_clips']} Filipino)")
    print(f"   Val:   {len(val_w)} clips, {stats['val']['total_hours']}h ({stats['val']['filipino_speech_clips']} Filipino)")
    print(f"   Test:  {len(test_w)} clips, {stats['test']['total_hours']}h ({stats['test']['filipino_speech_clips']} Filipino)")
    print(f"   Holdout: {len(holdout_w)} clips, {stats['holdout']['total_hours']}h ({stats['holdout']['filipino_speech_clips']} Filipino)")
    print("=" * 80)


if __name__ == "__main__":
    main()
