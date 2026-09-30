import os
import pandas as pd
import numpy as np
import soundfile as sf

BASE_DIR = "/home/misael.andre.maningo/MEng AI/AI 231/ME2 - Voice Command Model/upstream_repo/MEX2/OptionB"
manifest_path = os.path.join(BASE_DIR, "manifest.csv")

print(f"Loading manifest from: {manifest_path}")
df = pd.read_csv(manifest_path)

print("\n--- Manifest Overview ---")
print(f"Total rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

print("\n--- Splits Distribution ---")
split_counts = df.groupby("split")["speaker"].agg(["count", "nunique"])
split_counts.columns = ["file_count", "speaker_count"]
print(split_counts)

print("\n--- Speaker Distribution by Split ---")
for split_name, group in df.groupby("split"):
    speakers = sorted(group["speaker"].unique())
    print(f"{split_name}: {len(speakers)} speakers -> {speakers[:5]} ... {speakers[-5:]}")

print("\n--- Intent Distribution ---")
intent_counts = df["intent"].value_counts()
print(intent_counts)
print(f"Total distinct intents: {df['intent'].nunique()}")

print("\n--- Duration (seconds) Statistics ---")
durations = df["duration_sec"].dropna()
print(f"Count: {len(durations)}")
print(f"Min: {durations.min():.3f}s")
print(f"Mean: {durations.mean():.3f}s")
print(f"Median: {durations.median():.3f}s")
print(f"75th percentile: {np.percentile(durations, 75):.3f}s")
print(f"90th percentile: {np.percentile(durations, 90):.3f}s")
print(f"95th percentile: {np.percentile(durations, 95):.3f}s")
print(f"99th percentile: {np.percentile(durations, 99):.3f}s")
print(f"Max: {durations.max():.3f}s")

print("\n--- Checking Physical WAV Files (Sampling 200 files) ---")
sample_files = df["path"].sample(min(200, len(df)), random_state=42)
sample_rates = set()
channels = set()
missing_count = 0

for rel_path in sample_files:
    full_path = os.path.join(BASE_DIR, rel_path)
    if not os.path.exists(full_path):
        missing_count += 1
    else:
        try:
            info = sf.info(full_path)
            sample_rates.add(info.samplerate)
            channels.add(info.channels)
        except Exception as e:
            print(f"Error reading {full_path}: {e}")

print(f"Sample rates found: {sample_rates}")
print(f"Channels found: {channels}")
print(f"Missing files among sampled: {missing_count}")
