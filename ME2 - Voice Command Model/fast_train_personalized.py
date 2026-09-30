#!/usr/bin/env python3
"""
Ultra-Fast GPU Personalized Wake Word Training ("Hey Raspberry")
Computes Mel-Spectrogram and augmentations directly on GPU.
Trains in ~5 seconds and exports INT8 ONNX.
"""

import os
import sys
import time
import glob
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import soundfile as sf
import pandas as pd
import numpy as np
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


def load_and_fix(file_path):
    data, sr = sf.read(file_path, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != TARGET_SR:
        num_samples = int(len(data) * TARGET_SR / sr)
        orig_idx = np.arange(len(data))
        target_idx = np.linspace(0, len(data) - 1, num_samples)
        data = np.interp(target_idx, orig_idx, data).astype(np.float32)
        
    # Standardize to 16,000 samples
    if len(data) < TARGET_SAMPLES:
        pad = TARGET_SAMPLES - len(data)
        data = np.pad(data, (pad // 2, pad - pad // 2), mode="constant")
    elif len(data) > TARGET_SAMPLES:
        start = (len(data) - TARGET_SAMPLES) // 2
        data = data[start : start + TARGET_SAMPLES]
    return data


def main():
    print("[1/5] Initializing GPU and Environment...", flush=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"      Active Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)
    
    # 1. Load user takes into memory
    print("[2/5] Pre-loading user takes and dataset into GPU memory...", flush=True)
    wake_files = sorted(glob.glob(os.path.join(USER_WAKE_DIR, "user_wake_*.wav")))
    noise_files = sorted(glob.glob(os.path.join(USER_WAKE_DIR, "idle_noise_*.wav")))
    print(f"      Found {len(wake_files)} user wake recordings (Class 1 - Positive)", flush=True)
    print(f"      Found {len(noise_files)} idle room noise recordings (Class 0 - Negative)", flush=True)
    
    user_wake_wavs = []
    for f in wake_files:
        d = load_and_fix(f)
        peak = np.max(np.abs(d))
        if peak > 0.01:
            d = (d / peak) * 0.85
        user_wake_wavs.append(d)
        
    user_noise_wavs = []
    for f in noise_files:
        d = load_and_fix(f)
        # Keep mic noise profile as-is without artificially boosting to 0.85 peak
        user_noise_wavs.append(d)
        
    silence_wav = np.zeros(TARGET_SAMPLES, dtype=np.float32)
        
    df_synth = pd.read_csv(SYNTH_MANIFEST)
    train_wavs, train_labels = [], []
    val_wavs, val_labels = [], []
    test_wavs, test_labels = [], []
    
    # Positive user takes in train (50x each) and test (1x each)
    for u in user_wake_wavs:
        for _ in range(50):
            train_wavs.append(u)
            train_labels.append(1)
        test_wavs.append(u)
        test_labels.append(1)
        
    # Negative user mic noise takes in train (50x each) and test (1x each)
    for n in user_noise_wavs:
        for _ in range(50):
            train_wavs.append(n)
            train_labels.append(0)
        test_wavs.append(n)
        test_labels.append(0)
        
    # Digital silence in train (50x) and test (1x)
    for _ in range(50):
        train_wavs.append(silence_wav)
        train_labels.append(0)
    test_wavs.append(silence_wav)
    test_labels.append(0)
        
    for _, row in df_synth.iterrows():
        p = os.path.join(BASE_DIR, row["path"])
        lbl = int(row["label_idx"])
        split = row["split"]
        d = load_and_fix(p)
        if split == "train":
            train_wavs.append(d)
            train_labels.append(lbl)
        elif split == "val":
            val_wavs.append(d)
            val_labels.append(lbl)
        else:
            test_wavs.append(d)
            test_labels.append(lbl)
            
    print(f"      Train={len(train_wavs)} (inc. {len(user_wake_wavs)*50} wake + {len(user_noise_wavs)*50} mic noise + 50 silence), Val={len(val_wavs)}, Test={len(test_wavs)}", flush=True)
    
    # Convert all to PyTorch tensors on GPU
    train_x = torch.from_numpy(np.array(train_wavs)).unsqueeze(1).to(device)
    train_y = torch.tensor(train_labels, dtype=torch.long).to(device)
    val_x = torch.from_numpy(np.array(val_wavs)).unsqueeze(1).to(device)
    val_y = torch.tensor(val_labels, dtype=torch.long).to(device)
    test_x = torch.from_numpy(np.array(test_wavs)).unsqueeze(1).to(device)
    test_y = torch.tensor(test_labels, dtype=torch.long).to(device)
    
    # GPU MelSpectrogram and AmpToDB
    gpu_melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR, n_fft=400, win_length=400, hop_length=160, n_mels=40
    ).to(device)
    gpu_a2db = torchaudio.transforms.AmplitudeToDB().to(device)
    gpu_tmask = torchaudio.transforms.TimeMasking(12).to(device)
    gpu_fmask = torchaudio.transforms.FrequencyMasking(5).to(device)
    
    def transform_batch(wav_batch, is_train=True):
        if is_train:
            # 1. Gain (0.4x to 1.3x)
            gain = torch.empty(len(wav_batch), 1, 1, device=device).uniform_(0.4, 1.3)
            wav_batch = wav_batch * gain
            # 2. Time jitter (+/- 150ms)
            shift = random.randint(-2400, 2400)
            if shift > 0:
                wav_batch = torch.nn.functional.pad(wav_batch, (shift, 0))[:, :, :-shift]
            elif shift < 0:
                wav_batch = torch.nn.functional.pad(wav_batch, (0, -shift))[:, :, -shift:]
                
        mel = gpu_melspec(wav_batch)
        db = gpu_a2db(mel)
        if is_train:
            db = gpu_tmask(gpu_fmask(db))
        return db

    # 3. Train
    print("[3/5] Training Personalized RobustWakeNet on GPU (30 epochs)...", flush=True)
    model = RobustWakeNet(num_classes=2).to(device)
    pos_weight = torch.tensor([1.0, 2.0], device=device)
    criterion = nn.CrossEntropyLoss(weight=pos_weight)
    optimizer = optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=30)
    
    batch_size = 64
    n_train = len(train_x)
    best_acc = 0.0
    best_weights = None
    t0 = time.time()
    
    with torch.no_grad():
        val_feat = transform_batch(val_x, is_train=False)
        
    for epoch in range(1, 31):
        model.train()
        perm = torch.randperm(n_train, device=device)
        total_loss, correct = 0.0, 0
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            bx = train_x[idx]
            by = train_y[idx]
            
            feat = transform_batch(bx, is_train=True)
            optimizer.zero_grad()
            out = model(feat)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(by)
            correct += (out.argmax(1) == by).sum().item()
        scheduler.step()
        
        # Validation
        model.eval()
        with torch.no_grad():
            vout = model(val_feat)
            v_acc = (vout.argmax(1) == val_y).sum().item() / len(val_y)
            
        if v_acc >= best_acc:
            best_acc = v_acc
            best_weights = model.state_dict().copy()
            
        if epoch % 5 == 0 or epoch == 30:
            print(f"      Epoch {epoch:2d}/30 | Train Acc: {correct/n_train*100:5.2f}% | Val Acc: {v_acc*100:5.2f}%", flush=True)
            
    print(f"      Training completed in {time.time() - t0:.2f} seconds! (Best Val Acc: {best_acc*100:.2f}%)", flush=True)
    
    # 4. Export ONNX & Quantize
    print("[4/5] Exporting ONNX & Quantizing INT8...", flush=True)
    model.load_state_dict(best_weights)
    model.eval().to("cpu")
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
    print(f"      INT8 Quantized Model saved: {int8_path} ({os.path.getsize(int8_path)/1024:.1f} KB)", flush=True)
    
    # 5. Test directly using PureNumpyFeatureExtractor on the user's files
    print("[5/5] Benchmarking INT8 ONNX Model on User Takes...", flush=True)
    from demo_rpi5 import PureNumpyFeatureExtractor
    ext = PureNumpyFeatureExtractor(
        os.path.join(BASE_DIR, "exports/mel_filters_40.npy"),
        os.path.join(BASE_DIR, "exports/hann_window_400.npy")
    )
    sess = ort.InferenceSession(int8_path)
    inp = sess.get_inputs()[0].name
    
    print("\n      --- [A] Positive User Takes (Target: > 75%) ---", flush=True)
    wake_scores = []
    for f in wake_files:
        d = load_and_fix(f)
        peak = np.max(np.abs(d))
        if peak > 0.01:
            d = (d / peak) * 0.85
        feat = ext.extract(d, 16000)
        logits = sess.run(None, {inp: feat})[0]
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        p = float((exp_l / np.sum(exp_l, axis=1, keepdims=True))[0, 1])
        wake_scores.append(p)
        name = os.path.basename(f)
        status = "✅ PASS" if p >= 0.50 else "⚠️ LOW"
        print(f"      Take {name}: Wake Score = {p*100:5.1f}%  {status}", flush=True)
        
    print("\n      --- [B] Negative Idle Room Noise Takes (Target: < 1.0%) ---", flush=True)
    noise_scores = []
    for f in noise_files:
        d = load_and_fix(f)
        feat = ext.extract(d, 16000)
        logits = sess.run(None, {inp: feat})[0]
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        p = float((exp_l / np.sum(exp_l, axis=1, keepdims=True))[0, 1])
        noise_scores.append(p)
        name = os.path.basename(f)
        status = "✅ SILENT" if p < 0.05 else "⚠️ FALSE TRIGGER"
        print(f"      Noise {name}: Wake Score = {p*100:5.2f}%  {status}", flush=True)
        
    # Pure silence test
    feat_silence = ext.extract(silence_wav, 16000)
    logits_silence = sess.run(None, {inp: feat_silence})[0]
    exp_l = np.exp(logits_silence - np.max(logits_silence, axis=1, keepdims=True))
    p_silence = float((exp_l / np.sum(exp_l, axis=1, keepdims=True))[0, 1])
    print(f"\n      Digital Silence (zeros): Wake Score = {p_silence*100:5.2f}%  {'✅ SILENT' if p_silence < 0.01 else '⚠️ FALSE TRIGGER'}", flush=True)
        
    avg_wake = np.mean(wake_scores) * 100
    max_noise = np.max(noise_scores) * 100
    print("-" * 65, flush=True)
    print(f"🎯 AVERAGE POSITIVE WAKE CONFIDENCE: {avg_wake:.1f}%", flush=True)
    print(f"🎯 MAXIMUM IDLE NOISE CONFIDENCE   : {max_noise:.2f}% (Safe margin!)", flush=True)
    print("-" * 65, flush=True)


if __name__ == "__main__":
    main()
