import os
import json
import torch
import torchaudio
import soundfile as sf
import pandas as pd
import numpy as np

OPTIONB_DIR = "/home/misael.andre.maningo/MEng AI/AI 231/ME2 - Voice Command Model/upstream_repo/MEX2/OptionB"
MANIFEST_PATH = "/home/misael.andre.maningo/MEng AI/AI 231/ME2 - Voice Command Model/data/unified_manifest.csv"
LABELS_PATH = "/home/misael.andre.maningo/MEng AI/AI 231/ME2 - Voice Command Model/data/labels_20.json"

TARGET_SR = 16000
TARGET_SAMPLES = 32000 # 2.0 seconds

class VoiceCommandDataset(torch.utils.data.Dataset):
    def __init__(self, manifest_df, base_dir, target_sr=16000, target_samples=32000, is_train=False):
        self.df = manifest_df.reset_index(drop=True)
        self.base_dir = base_dir
        self.target_sr = target_sr
        self.target_samples = target_samples
        self.is_train = is_train
        
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=target_sr,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=40
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.base_dir, row["path"])
        
        # Robust loading with soundfile (supports WAV 12kHz, 16kHz, etc.)
        data, sr = sf.read(file_path, dtype="float32")
        waveform = torch.from_numpy(data)
        
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0) # (1, T)
        else:
            waveform = waveform.t().mean(dim=0, keepdim=True) # mono (1, T)
        
        # Resample to 16kHz if needed
        if sr != self.target_sr:
            resampler = torchaudio.transforms.Resample(sr, self.target_sr)
            waveform = resampler(waveform)
            
        # Fit to target length (32,000 samples = 2.0s)
        length = waveform.shape[-1]
        if length < self.target_samples:
            pad_amount = self.target_samples - length
            pad_left = pad_amount // 2
            pad_right = pad_amount - pad_left
            waveform = torch.nn.functional.pad(waveform, (pad_left, pad_right))
        elif length > self.target_samples:
            start = (length - self.target_samples) // 2
            waveform = waveform[:, start : start + self.target_samples]
            
        # Extract Log-Mel Spectrogram: (1, n_mels=40, time=201)
        mel = self.melspec(waveform)
        log_mel = self.amp_to_db(mel)
        
        label = int(row["label_idx"])
        return log_mel, label

df = pd.read_csv(MANIFEST_PATH)
print("Testing DataLoader on Train, Val, and Test splits...")

for split_name in ["train", "val", "test"]:
    split_df = df[df["split"] == split_name]
    dataset = VoiceCommandDataset(split_df, OPTIONB_DIR, is_train=(split_name == "train"))
    loader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2)
    
    batch_x, batch_y = next(iter(loader))
    print(f"[{split_name.upper()}] Batch X shape: {batch_x.shape}, Batch Y shape: {batch_y.shape}, Range: [{batch_x.min():.1f}, {batch_x.max():.1f}] dB")

print("All splits loaded and verified successfully!")
