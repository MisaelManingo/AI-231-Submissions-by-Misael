import os
import glob
import random
import asyncio
import numpy as np
import pandas as pd
import soundfile as sf
import librosa
import edge_tts
from tqdm import tqdm

TARGET_SR = 16000
TARGET_DURATION = 1.0 # 1.0 second window for wake word
TARGET_SAMPLES = int(TARGET_SR * TARGET_DURATION)

BASE_DIR = "/home/misael.andre.maningo/MEng AI/AI 231/ME2 - Voice Command Model"
OUTPUT_DIR = os.path.join(BASE_DIR, "data/wakeword")
WAVS_DIR = os.path.join(OUTPUT_DIR, "wavs")
MANIFEST_OUT = os.path.join(OUTPUT_DIR, "wakeword_manifest.csv")
OPTIONB_DIR = os.path.join(BASE_DIR, "upstream_repo/MEX2/OptionB")
RAW_NEG_DIR = os.path.join(BASE_DIR, "data/raw_negatives")

os.makedirs(WAVS_DIR, exist_ok=True)
random.seed(42)
np.random.seed(42)

# Diverse English voices (including Filipino-English en-PH)
VOICES = [
    "en-US-AriaNeural",
    "en-US-GuyNeural",
    "en-US-JennyNeural",
    "en-US-ChristopherNeural",
    "en-US-MichelleNeural",
    "en-US-EricNeural",
    "en-PH-RosaNeural",
    "en-PH-JamesNeural",
    "en-GB-SoniaNeural",
    "en-GB-RyanNeural",
    "en-AU-NatashaNeural",
    "en-CA-ClaraNeural",
    "en-IN-NeerjaNeural"
]

RATES = ["-15%", "-10%", "+0%", "+10%", "+15%"]
PITCHES = ["-10Hz", "+0Hz", "+10Hz"]

# Helper to normalize audio to exactly 1.0s @ 16kHz
def fit_audio_1s(y, sr):
    if sr != TARGET_SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=TARGET_SR)
    if len(y) < TARGET_SAMPLES:
        pad = TARGET_SAMPLES - len(y)
        pad_l = pad // 2
        pad_r = pad - pad_l
        y = np.pad(y, (pad_l, pad_r), mode="constant")
    elif len(y) > TARGET_SAMPLES:
        start = (len(y) - TARGET_SAMPLES) // 2
        y = y[start : start + TARGET_SAMPLES]
    # Normalize peak
    max_val = np.max(np.abs(y))
    if max_val > 1e-4:
        y = y / max_val * 0.95
    return y

async def generate_speech(text, voice, rate, pitch, out_path):
    communicate = edge_tts.Communicate(text, voice, rate=rate, pitch=pitch)
    temp_mp3 = out_path.replace(".wav", "_temp.mp3")
    await communicate.save(temp_mp3)
    y, sr = librosa.load(temp_mp3, sr=TARGET_SR)
    y_1s = fit_audio_1s(y, sr)
    sf.write(out_path, y_1s, TARGET_SR)
    if os.path.exists(temp_mp3):
        os.remove(temp_mp3)

async def main():
    print("=" * 60)
    print("GENERATING WAKE WORD DATASET FOR 'Hey Raspberry'")
    print("=" * 60)
    
    samples = []
    
    # 1. POSITIVES: "Hey Raspberry"
    print("Generating positive samples ('Hey Raspberry')...")
    pos_count = 0
    for voice in VOICES:
        for rate in RATES:
            for pitch in PITCHES:
                out_name = f"pos_hey_raspberry_{voice}_{rate}_{pitch}.wav".replace("%", "pct").replace("+", "p").replace("-", "m")
                out_path = os.path.join(WAVS_DIR, out_name)
                try:
                    await generate_speech("Hey Raspberry", voice, rate, pitch, out_path)
                    samples.append({
                        "path": os.path.relpath(out_path, BASE_DIR),
                        "label": "WAKE_WORD",
                        "label_idx": 1,
                        "type": "positive",
                        "speaker": voice
                    })
                    pos_count += 1
                except Exception as e:
                    print(f"Error generating positive {out_name}: {e}")
                    
    print(f"Generated {pos_count} positive 'Hey Raspberry' utterances.")
    
    # 2. CONFUSERS (Near-Miss Negative Speech)
    CONFUSERS = [
        "Raspberry",
        "Hey Blackberry",
        "Hey Strawberry",
        "Hey Siri",
        "Hey Robot",
        "Hey Computer",
        "Blueberry",
        "Raspberry Pi",
        "Strawberry"
    ]
    print("Generating phonetically similar confusers...")
    conf_count = 0
    # Sample a diverse subset across voices
    for conf_text in CONFUSERS:
        selected_voices = random.sample(VOICES, 8)
        for voice in selected_voices:
            rate = random.choice(RATES)
            pitch = random.choice(PITCHES)
            safe_text = conf_text.lower().replace(" ", "_")
            out_name = f"neg_confuser_{safe_text}_{voice}_{rate}_{pitch}.wav".replace("%", "pct").replace("+", "p").replace("-", "m")
            out_path = os.path.join(WAVS_DIR, out_name)
            try:
                await generate_speech(conf_text, voice, rate, pitch, out_path)
                samples.append({
                    "path": os.path.relpath(out_path, BASE_DIR),
                    "label": "NOT_WAKE",
                    "label_idx": 0,
                    "type": "confuser",
                    "speaker": voice
                })
                conf_count += 1
            except Exception as e:
                print(f"Error generating confuser {out_name}: {e}")
                
    print(f"Generated {conf_count} confuser negative utterances.")
    
    # 3. NON-WAKE SAMPLES FROM OPTION B COMMANDS
    print("Extracting 1.0s negative samples from Option B commands...")
    opt_df = pd.read_csv(os.path.join(OPTIONB_DIR, "manifest.csv"))
    opt_sample = opt_df.sample(n=350, random_state=42)
    opt_count = 0
    for idx, row in opt_sample.iterrows():
        src_path = os.path.join(OPTIONB_DIR, row["path"])
        if os.path.exists(src_path):
            data, sr = sf.read(src_path, dtype="float32")
            y_1s = fit_audio_1s(data, sr)
            out_name = f"neg_cmd_{os.path.basename(row['path'])}"
            out_path = os.path.join(WAVS_DIR, out_name)
            sf.write(out_path, y_1s, TARGET_SR)
            samples.append({
                "path": os.path.relpath(out_path, BASE_DIR),
                "label": "NOT_WAKE",
                "label_idx": 0,
                "type": "command_negative",
                "speaker": row["speaker"]
            })
            opt_count += 1
    print(f"Extracted {opt_count} command negative samples.")
    
    # 4. AMBIENT NOISE & SILENCE
    print("Extracting ambient noise & silence negatives...")
    bg_files = glob.glob(os.path.join(RAW_NEG_DIR, "_background_noise_", "*.wav"))
    noise_count = 0
    for bg_path in bg_files:
        y, sr = librosa.load(bg_path, sr=TARGET_SR)
        hop = TARGET_SAMPLES
        for s_idx in range(min(25, (len(y) - TARGET_SAMPLES) // hop)):
            chunk = y[s_idx * hop : (s_idx + 1) * hop]
            out_name = f"neg_noise_{os.path.splitext(os.path.basename(bg_path))[0]}_s{s_idx:02d}.wav"
            out_path = os.path.join(WAVS_DIR, out_name)
            sf.write(out_path, chunk, TARGET_SR)
            samples.append({
                "path": os.path.relpath(out_path, BASE_DIR),
                "label": "NOT_WAKE",
                "label_idx": 0,
                "type": "ambient_noise",
                "speaker": "ambient"
            })
            noise_count += 1
    print(f"Extracted {noise_count} ambient noise slices.")
    
    # Create DataFrame and Split
    df = pd.DataFrame(samples)
    print(f"\nTotal Wake Word Samples: {len(df)}")
    print(df["label"].value_counts())
    print(df["type"].value_counts())
    
    # Stratified Split: 80% train, 10% val, 10% test
    # Ensure speaker / voice disjointness for validation & test sets
    pos_mask = df["label_idx"] == 1
    neg_mask = df["label_idx"] == 0
    
    pos_df = df[pos_mask].sample(frac=1.0, random_state=42).reset_index(drop=True)
    neg_df = df[neg_mask].sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    def assign_split(sub_df):
        n = len(sub_df)
        n_val = int(0.10 * n)
        n_test = int(0.10 * n)
        splits = ["val"] * n_val + ["test"] * n_test + ["train"] * (n - n_val - n_test)
        sub_df["split"] = splits
        return sub_df

    pos_df = assign_split(pos_df)
    neg_df = assign_split(neg_df)
    
    final_df = pd.concat([pos_df, neg_df], ignore_index=True).sample(frac=1.0, random_state=42).reset_index(drop=True)
    print("\nSplit Distribution:")
    print(final_df.groupby(["split", "label"]).size())
    
    final_df.to_csv(MANIFEST_OUT, index=False)
    print(f"\nSaved wake word manifest to: {MANIFEST_OUT}")

if __name__ == "__main__":
    asyncio.run(main())
