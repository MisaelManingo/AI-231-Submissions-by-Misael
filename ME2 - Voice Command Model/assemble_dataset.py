import os
import json
import glob
import random
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
from tqdm import tqdm

TARGET_SR = 16000
TARGET_DURATION = 2.0
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)

OPTIONB_DIR = "/home/misael.andre.maningo/MEng AI/AI 231/ME2 - Voice Command Model/upstream_repo/MEX2/OptionB"
RAW_NEG_DIR = "/home/misael.andre.maningo/MEng AI/AI 231/ME2 - Voice Command Model/data/raw_negatives"
OUTPUT_NEG_DIR = "/home/misael.andre.maningo/MEng AI/AI 231/ME2 - Voice Command Model/data/negatives_2s"
OUT_MANIFEST = "/home/misael.andre.maningo/MEng AI/AI 231/ME2 - Voice Command Model/data/unified_manifest.csv"
OUT_LABELS = "/home/misael.andre.maningo/MEng AI/AI 231/ME2 - Voice Command Model/data/labels_20.json"

os.makedirs(OUTPUT_NEG_DIR, exist_ok=True)
random.seed(42)
np.random.seed(42)

print("--- Step 1: Loading Option B Manifest ---")
df_opt = pd.read_csv(os.path.join(OPTIONB_DIR, "manifest.csv"))
print(f"Option B rows: {len(df_opt)}")

# 19 positive intents
with open(os.path.join(OPTIONB_DIR, "labels.json"), "r") as f:
    positive_labels = json.load(f)

# Define 20-class label map with _BACKGROUND_ as 0
label2idx = {"_BACKGROUND_": 0}
for i, lbl in enumerate(positive_labels):
    label2idx[lbl] = i + 1

idx2label = {v: k for k, v in label2idx.items()}
with open(OUT_LABELS, "w") as f:
    json.dump({"label2idx": label2idx, "idx2label": idx2label}, f, indent=2)
print(f"Created label mapping for 20 classes. Saved to {OUT_LABELS}")

print("\n--- Step 2: Processing Negative / Background Data ---")
neg_samples = []

# Helper function to pad/crop to exactly TARGET_SAMPLES
def fit_to_target_length(y, sr):
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    if len(y) < TARGET_SAMPLES:
        # Center audio and pad with silence
        pad_total = TARGET_SAMPLES - len(y)
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        y = np.pad(y, (pad_left, pad_right), mode="constant")
    elif len(y) > TARGET_SAMPLES:
        # Center crop
        start = (len(y) - TARGET_SAMPLES) // 2
        y = y[start : start + TARGET_SAMPLES]
    return y

# A. Slice continuous background noises
bg_files = glob.glob(os.path.join(RAW_NEG_DIR, "_background_noise_", "*.wav"))
print(f"Found {len(bg_files)} continuous background noise files.")

bg_slices = []
for bg_path in bg_files:
    fname = os.path.splitext(os.path.basename(bg_path))[0]
    y, sr = librosa.load(bg_path, sr=TARGET_SR, mono=True)
    hop = int(TARGET_SR * 1.0) # 1.0s hop = 50% overlap for diversity
    num_slices = (len(y) - TARGET_SAMPLES) // hop
    for s_idx in range(num_slices):
        start = s_idx * hop
        chunk = y[start : start + TARGET_SAMPLES]
        out_name = f"bg_{fname}_slice{s_idx:03d}.wav"
        out_path = os.path.join(OUTPUT_NEG_DIR, out_name)
        sf.write(out_path, chunk, TARGET_SR)
        bg_slices.append({
            "rel_path": os.path.relpath(out_path, OPTIONB_DIR),
            "full_path": out_path,
            "source": f"bg_{fname}"
        })

print(f"Generated {len(bg_slices)} slices from continuous background noise.")

# B. Process room silence / ambient files
silence_files = sorted(glob.glob(os.path.join(RAW_NEG_DIR, "_silence_", "*.wav")))
print(f"Found {len(silence_files)} silence/room noise files.")
silence_slices = []
for s_path in silence_files:
    fname = os.path.splitext(os.path.basename(s_path))[0]
    y, sr = librosa.load(s_path, sr=TARGET_SR, mono=True)
    # Loop / tile short silence to 2s
    if len(y) < TARGET_SAMPLES:
        repeats = int(np.ceil(TARGET_SAMPLES / len(y)))
        y = np.tile(y, repeats)[:TARGET_SAMPLES]
    else:
        y = y[:TARGET_SAMPLES]
    out_name = f"silence_{fname}.wav"
    out_path = os.path.join(OUTPUT_NEG_DIR, out_name)
    sf.write(out_path, y, TARGET_SR)
    silence_slices.append({
        "rel_path": os.path.relpath(out_path, OPTIONB_DIR),
        "full_path": out_path,
        "source": "silence"
    })
print(f"Processed {len(silence_slices)} silence clips.")

# C. Process unknown/non-command speech words
unknown_files = sorted(glob.glob(os.path.join(RAW_NEG_DIR, "_unknown_", "*.wav")))
print(f"Found {len(unknown_files)} unknown speech files.")
unknown_slices = []
for u_path in unknown_files:
    fname = os.path.splitext(os.path.basename(u_path))[0]
    y, sr = librosa.load(u_path, sr=TARGET_SR, mono=True)
    y_fitted = fit_to_target_length(y, sr)
    out_name = f"unknown_{fname}.wav"
    out_path = os.path.join(OUTPUT_NEG_DIR, out_name)
    sf.write(out_path, y_fitted, TARGET_SR)
    unknown_slices.append({
        "rel_path": os.path.relpath(out_path, OPTIONB_DIR),
        "full_path": out_path,
        "source": "unknown_speech"
    })
print(f"Processed {len(unknown_slices)} unknown speech clips.")

# Combine all negatives
all_negatives = bg_slices + silence_slices + unknown_slices
print(f"Total available negative samples: {len(all_negatives)}")

# Shuffle with fixed seed and split: 80% train, 10% val, 10% test
random.shuffle(all_negatives)
n_total = len(all_negatives)
n_val = int(0.10 * n_total)
n_test = int(0.10 * n_total)
n_train = n_total - n_val - n_test

val_neg = all_negatives[:n_val]
test_neg = all_negatives[n_val : n_val + n_test]
train_neg = all_negatives[n_val + n_test :]

print(f"Negative splits -> Train: {len(train_neg)}, Val: {len(val_neg)}, Test: {len(test_neg)}")

neg_rows = []
for item in train_neg:
    neg_rows.append({
        "path": item["rel_path"],
        "label": "_BACKGROUND_",
        "intent": "_BACKGROUND_",
        "speaker": f"neg_{item['source']}",
        "split": "train",
        "phrase_id": "neg",
        "variant_id": "neg",
        "transcript": "[BACKGROUND_OR_NON_COMMAND]",
        "slot": "",
        "slot_value": "",
        "duration_sec": TARGET_DURATION,
        "label_idx": 0
    })

for item in val_neg:
    neg_rows.append({
        "path": item["rel_path"],
        "label": "_BACKGROUND_",
        "intent": "_BACKGROUND_",
        "speaker": f"neg_{item['source']}",
        "split": "val",
        "phrase_id": "neg",
        "variant_id": "neg",
        "transcript": "[BACKGROUND_OR_NON_COMMAND]",
        "slot": "",
        "slot_value": "",
        "duration_sec": TARGET_DURATION,
        "label_idx": 0
    })

for item in test_neg:
    neg_rows.append({
        "path": item["rel_path"],
        "label": "_BACKGROUND_",
        "intent": "_BACKGROUND_",
        "speaker": f"neg_{item['source']}",
        "split": "test",
        "phrase_id": "neg",
        "variant_id": "neg",
        "transcript": "[BACKGROUND_OR_NON_COMMAND]",
        "slot": "",
        "slot_value": "",
        "duration_sec": TARGET_DURATION,
        "label_idx": 0
    })

df_neg = pd.DataFrame(neg_rows)

print("\n--- Step 3: Merging with Option B Manifest ---")
df_opt["label_idx"] = df_opt["intent"].map(label2idx)

# Align columns
unified_df = pd.concat([df_opt, df_neg], ignore_index=True)
print(f"Total unified rows: {len(unified_df)}")

# Summary by split and label
print("\n--- Unified Split Distribution ---")
print(unified_df.groupby(["split"])["label_idx"].agg(["count"]))

print("\n--- Class Counts in Unified Dataset ---")
print(unified_df["intent"].value_counts())

# Save unified manifest
unified_df.to_csv(OUT_MANIFEST, index=False)
print(f"\nSuccessfully wrote unified manifest to: {OUT_MANIFEST}")
