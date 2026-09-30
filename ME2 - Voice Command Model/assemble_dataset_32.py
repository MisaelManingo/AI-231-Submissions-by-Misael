#!/usr/bin/env python3
"""
Assemble 32-Class Joint Intent & Slot Dataset for ME2
Combines 31 Option B command/slot categories + 1 _BACKGROUND_ class.
Preserves speaker-disjoint splits (80/10/10) and creates labels_32.json with slot metadata.
"""

import os
import json
import glob
import random
import numpy as np
import pandas as pd

BASE_DIR = "/home/misael.andre.maningo/MEng AI/AI 231/ME2 - Voice Command Model"
OPTIONB_DIR = os.path.join(BASE_DIR, "upstream_repo/MEX2/OptionB")
NEG_DIR = os.path.join(BASE_DIR, "data/negatives_2s")
OUT_MANIFEST = os.path.join(BASE_DIR, "data/unified_manifest_32.csv")
OUT_LABELS = os.path.join(BASE_DIR, "data/labels_32.json")

random.seed(42)
np.random.seed(42)

print("--- Step 1: Loading Option B Manifest ---")
df_opt = pd.read_csv(os.path.join(OPTIONB_DIR, "manifest.csv"))
print(f"Option B rows: {len(df_opt)}")

# Extract 31 unique command/slot labels
unique_labels = sorted(df_opt["label"].unique().tolist())
print(f"Found {len(unique_labels)} unique command/slot labels.")

# Define 32-class label mapping with _BACKGROUND_ as 0
label2idx = {"_BACKGROUND_": 0}
for i, lbl in enumerate(unique_labels):
    label2idx[lbl] = i + 1

idx2label = {v: k for k, v in label2idx.items()}

# Build metadata map (intent, slot, slot_value) for fast parsing
slot_meta = {}
slot_meta["_BACKGROUND_"] = {"intent": "_BACKGROUND_", "slot": None, "slot_value": None}
for _, row in df_opt[["label", "intent", "slot", "slot_value"]].drop_duplicates().iterrows():
    lbl = row["label"]
    slot_meta[lbl] = {
        "intent": str(row["intent"]) if pd.notna(row["intent"]) else lbl,
        "slot": str(row["slot"]) if pd.notna(row["slot"]) else None,
        "slot_value": str(row["slot_value"]) if pd.notna(row["slot_value"]) else None,
    }

labels_payload = {
    "num_classes": len(label2idx),
    "label2idx": label2idx,
    "idx2label": idx2label,
    "slot_meta": slot_meta
}

with open(OUT_LABELS, "w") as f:
    json.dump(labels_payload, f, indent=2)
print(f"Created 32-class label map with slot metadata -> {OUT_LABELS}")

# Assign label_idx in df_opt based on the joint 'label' column
df_opt["label_idx"] = df_opt["label"].map(label2idx)

print("\n--- Step 2: Loading Background / Negative Audio ---")
# Use the pre-sliced 2.0s negative clips from data/negatives_2s
neg_files = sorted(glob.glob(os.path.join(NEG_DIR, "*.wav")))
print(f"Found {len(neg_files)} pre-sliced 2.0s negative clips in {NEG_DIR}")

# Split negatives 80% train, 10% val, 10% test
random.shuffle(neg_files)
n_total = len(neg_files)
n_val = int(0.10 * n_total)
n_test = int(0.10 * n_total)
n_train = n_total - n_val - n_test

val_neg = neg_files[:n_val]
test_neg = neg_files[n_val : n_val + n_test]
train_neg = neg_files[n_val + n_test :]

neg_rows = []
for split_name, flist in [("train", train_neg), ("val", val_neg), ("test", test_neg)]:
    for p in flist:
        rel_p = os.path.relpath(p, OPTIONB_DIR)
        neg_rows.append({
            "path": rel_p,
            "label": "_BACKGROUND_",
            "intent": "_BACKGROUND_",
            "speaker": f"neg_{os.path.basename(p)[:12]}",
            "split": split_name,
            "phrase_id": "neg",
            "variant_id": "neg",
            "transcript": "[BACKGROUND_OR_NON_COMMAND]",
            "slot": "",
            "slot_value": "",
            "duration_sec": 2.0,
            "label_idx": 0
        })

df_neg = pd.DataFrame(neg_rows)
print(f"Negative splits -> Train: {len(train_neg)}, Val: {len(val_neg)}, Test: {len(test_neg)}")

print("\n--- Step 3: Merging Option B + Negatives ---")
unified_df = pd.concat([df_opt, df_neg], ignore_index=True)
print(f"Total unified rows: {len(unified_df)}")

print("\n--- Split Distribution ---")
print(unified_df.groupby(["split"])["label_idx"].agg(["count"]))

unified_df.to_csv(OUT_MANIFEST, index=False)
print(f"\nSuccessfully wrote unified 32-class manifest to: {OUT_MANIFEST}")
