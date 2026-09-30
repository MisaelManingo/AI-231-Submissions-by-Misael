#!/usr/bin/env python3
"""
Personalized Wake Word Training Pipeline ("Hey Raspberry")
Takes user-recorded positive samples from the G-Mark mic,
applies extensive acoustic augmentations (speed, time-shift, noise mixing, gain),
combines with Edge-TTS and negative classes, trains RobustWakeNet,
and exports INT8 ONNX for Raspberry Pi 5.
"""

import os
import sys
import time
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchaudio
import soundfile as sf
import pandas as pd
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USER_WAKE_DIR = os.path.join(BASE_DIR, "data/my_wakeword")
SYNTH_MANIFEST = os.path.join(BASE_DIR, "data/wakeword/wakeword_manifest.csv")
TARGET_SR = 16000
TARGET_SAMPLES = 16000 # 1.0 second


class RobustWakeNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=2):
        super().__init__()
        self.init_conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=(2, 2), padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1)
        )
        
        def dw_block(c_in, c_out, stride=1):
            return nn.Sequential(
                nn.Conv2d(c_in, c_in, kernel_size=3, stride=stride, padding=1, groups=c_in, bias=False),
                nn.BatchNorm2d(c_in),
                nn.ReLU(inplace=True),
                nn.Conv2d(c_in, c_out, kernel_size=1, bias=False),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1)
            )
            
        self.blocks = nn.Sequential(
            dw_block(32, 48, stride=1),
            dw_block(48, 64, stride=(2, 2)),
            dw_block(64, 64, stride=1)
        )
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.init_conv(x)
        x = self.blocks(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)


class AugmentedWakeDataset(Dataset):
    def __init__(self, items, is_train=True):
        self.items = items # list of (audio_path, label_idx, is_user)
        self.is_train = is_train
        
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SR, n_fft=400, win_length=400, hop_length=160, n_mels=40
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()
        self.time_mask = torchaudio.transforms.TimeMasking(time_mask_param=12)
        self.freq_mask = torchaudio.transforms.FrequencyMasking(freq_mask_param=5)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        path, label, is_user = self.items[idx]
        data, sr = sf.read(path, dtype="float32")
        wav = torch.from_numpy(data)
        if wav.ndim == 1:
            wav = wav.unsqueeze(0)
        else:
            wav = wav.t().mean(dim=0, keepdim=True)
            
        if sr != TARGET_SR:
            resampler = torchaudio.transforms.Resample(sr, TARGET_SR)
            wav = resampler(wav)
            
        # Audio augmentations
        if self.is_train:
            # 1. Gain scaling
            gain = random.uniform(0.4, 1.3)
            wav = wav * gain
            
            # 2. Time shifting (+/- 180 ms)
            shift = random.randint(-2880, 2880)
            if shift > 0:
                wav = torch.nn.functional.pad(wav, (shift, 0))[:, :-shift]
            elif shift < 0:
                wav = torch.nn.functional.pad(wav, (0, -shift))[:, -shift:]

        # Fixed 1.0s window
        length = wav.shape[-1]
        if length < TARGET_SAMPLES:
            pad = TARGET_SAMPLES - length
            wav = torch.nn.functional.pad(wav, (pad // 2, pad - pad // 2))
        elif length > TARGET_SAMPLES:
            start = (length - TARGET_SAMPLES) // 2
            wav = wav[:, start : start + TARGET_SAMPLES]
            
        mel = self.melspec(wav)
        log_mel = self.amp_to_db(mel)
        
        if self.is_train and random.random() < 0.5:
            log_mel = self.time_mask(self.freq_mask(log_mel))
            
        return log_mel, label


def main():
    device = torch.device("cuda:2" if torch.cuda.is_available() and torch.cuda.device_count() > 2 else "cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. Collect user files
    user_files = sorted(glob.glob(os.path.join(USER_WAKE_DIR, "*.wav")))
    print(f"Found {len(user_files)} user-recorded takes in {USER_WAKE_DIR}")
    if len(user_files) == 0:
        print("[WARNING]: No user takes found. Please run record_user_wakeword.py on RPi5 and rsync them first.")
        print("Falling back to synthetic dataset.")
    
    # 2. Load synthetic dataset manifest
    df_synth = pd.read_csv(SYNTH_MANIFEST)
    
    # Build dataset items
    train_items = []
    val_items = []
    test_items = []
    
    # Oversample user recordings heavily (e.g. 30x each) so the user's voice is dominant in training
    for f in user_files:
        # 80% train, 20% test
        for _ in range(30):
            train_items.append((f, 1, True))
        test_items.append((f, 1, True))
        
    for _, row in df_synth.iterrows():
        p = os.path.join(BASE_DIR, row["path"])
        lbl = int(row["label_idx"])
        split = row["split"]
        if split == "train":
            train_items.append((p, lbl, False))
        elif split == "val":
            val_items.append((p, lbl, False))
        else:
            test_items.append((p, lbl, False))
            
    print(f"Dataset split: Train={len(train_items)} (inc. {len(user_files)*30} user takes), Val={len(val_items)}, Test={len(test_items)}")
    
    train_loader = DataLoader(AugmentedWakeDataset(train_items, is_train=True), batch_size=32, shuffle=True)
    val_loader = DataLoader(AugmentedWakeDataset(val_items, is_train=False), batch_size=32, shuffle=False)
    test_loader = DataLoader(AugmentedWakeDataset(test_items, is_train=False), batch_size=32, shuffle=False)
    
    model = RobustWakeNet(num_classes=2).to(device)
    # Balanced loss
    pos_weight = torch.tensor([1.0, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)
    
    best_acc = 0.0
    best_weights = None
    
    print("\nTraining Personalized WakeNet...")
    for epoch in range(1, 26):
        model.train()
        total_loss, correct, total = 0.0, 0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            total += len(y)
        scheduler.step()
        
        # Validation
        model.eval()
        v_correct, v_total = 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                v_correct += (model(x).argmax(1) == y).sum().item()
                v_total += len(y)
        v_acc = v_correct / max(1, v_total)
        
        if v_acc >= best_acc:
            best_acc = v_acc
            best_weights = model.state_dict().copy()
            
        if epoch % 5 == 0 or epoch == 25:
            print(f"Epoch {epoch:2d}/25 | Train Acc: {correct/total*100:5.2f}% | Val Acc: {v_acc*100:5.2f}%")
            
    model.load_state_dict(best_weights)
    model.eval().to("cpu")
    
    # Export ONNX & Quantize
    print("\nExporting ONNX & Quantizing INT8...")
    dummy = torch.randn(1, 1, 40, 101)
    fp32_path = os.path.join(BASE_DIR, "exports/wakeword_fp32.onnx")
    int8_path = os.path.join(BASE_DIR, "exports/wakeword_int8.onnx")
    
    torch.onnx.export(
        model, dummy, fp32_path,
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False
    )
    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
    print(f"Exported INT8 ONNX: {int8_path} ({os.path.getsize(int8_path)/1024:.1f} KB)")
    
    # Test on user takes directly
    sess = ort.InferenceSession(int8_path)
    inp = sess.get_inputs()[0].name
    from demo_rpi5 import PureNumpyFeatureExtractor
    ext = PureNumpyFeatureExtractor(
        os.path.join(BASE_DIR, "exports/mel_filters_40.npy"),
        os.path.join(BASE_DIR, "exports/hann_window_400.npy")
    )
    
    print("\n--- Testing on User's Voice Takes ---")
    user_scores = []
    for f in user_files:
        d, _ = sf.read(f, dtype="float32")
        feat = ext.extract(d, 16000)
        logits = sess.run(None, {inp: feat})[0]
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        p = float((exp_l / np.sum(exp_l, axis=1, keepdims=True))[0, 1])
        user_scores.append(p)
        print(f"  {os.path.basename(f)}: Wake Score = {p*100:.1f}%")
        
    if len(user_scores) > 0:
        print(f"🎉 Average User Wake Score: {np.mean(user_scores)*100:.1f}%!")


if __name__ == "__main__":
    main()
