#!/usr/bin/env python3
"""
Ultra-Fast GPU Training for 32-Class Joint Intent & Slot Voice Command Model (VCM)
Architecture: BC-ResNet (69,696 parameters, ~70 KB INT8)
Pre-loads audio directly to memory, runs MelSpectrogram & SpecAugment on GPU.
Trains 25 epochs in ~1-2 minutes on NVIDIA A100.
"""

import os
import sys
import time
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import soundfile as sf
import pandas as pd
import numpy as np
import glob
import concurrent.futures
from sklearn.metrics import accuracy_score, f1_score, classification_report
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

from models.bcresnet import get_bcresnet

BASE_DIR = "/home/misael.andre.maningo/MEng AI/AI 231/ME2 - Voice Command Model"
OPTIONB_DIR = os.path.join(BASE_DIR, "upstream_repo/MEX2/OptionB")
MANIFEST_PATH = os.path.join(BASE_DIR, "data/unified_manifest_32.csv")
LABELS_PATH = os.path.join(BASE_DIR, "data/labels_32.json")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports")
os.makedirs(EXPORTS_DIR, exist_ok=True)

TARGET_SR = 16000
TARGET_SAMPLES = 32000 # 2.0 seconds


def load_audio_sample(file_path):
    # Robust read via soundfile
    data, sr = sf.read(file_path, dtype="float32")
    if data.ndim > 1:
        data = np.mean(data, axis=1)
    if sr != TARGET_SR:
        num_samples = int(len(data) * TARGET_SR / sr)
        orig_idx = np.arange(len(data))
        target_idx = np.linspace(0, len(data) - 1, num_samples)
        data = np.interp(target_idx, orig_idx, data).astype(np.float32)
        
    if len(data) < TARGET_SAMPLES:
        pad = TARGET_SAMPLES - len(data)
        data = np.pad(data, (pad // 2, pad - pad // 2), mode="constant")
    elif len(data) > TARGET_SAMPLES:
        start = (len(data) - TARGET_SAMPLES) // 2
        data = data[start : start + TARGET_SAMPLES]
    return data


def main():
    print("=" * 65, flush=True)
    print("🚀 FAST 32-CLASS JOINT INTENT & SLOT VCM TRAINING (BC-RESNET)", flush=True)
    print("=" * 65, flush=True)

    # 1. Setup Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[1/6] Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})", flush=True)

    # 2. Load Label Map
    with open(LABELS_PATH, "r") as f:
        label_info = json.load(f)
    num_classes = label_info["num_classes"]
    idx2label = {int(k): v for k, v in label_info["idx2label"].items()}
    class_names = [idx2label[i] for i in range(num_classes)]
    print(f"      Loaded {num_classes} classes from {LABELS_PATH}", flush=True)

    # 3. Preload all audio into memory
    print("[2/6] Loading unified 32-class dataset into memory...", flush=True)
    df = pd.read_csv(MANIFEST_PATH)
    t0_load = time.time()
    
    train_df = df[df["split"] == "train"].reset_index(drop=True)
    val_df = df[df["split"] == "val"].reset_index(drop=True)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"      Splits: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}", flush=True)

    def load_split_arrays(split_df):
        wavs = np.zeros((len(split_df), TARGET_SAMPLES), dtype=np.float32)
        labels = np.zeros(len(split_df), dtype=np.int64)
        for i, row in split_df.iterrows():
            p = os.path.join(OPTIONB_DIR, row["path"])
            wavs[i] = load_audio_sample(p)
            labels[i] = int(row["label_idx"])
        paths = [os.path.join(OPTIONB_DIR, p) for p in split_df["path"]]
        labels = split_df["label_idx"].values.astype(np.int64)
        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            wavs_list = list(executor.map(load_audio_sample, paths))
        wavs = np.array(wavs_list, dtype=np.float32)
        return wavs, labels

    train_wavs, train_labels = load_split_arrays(train_df)
    val_wavs, val_labels = load_split_arrays(val_df)
    test_wavs, test_labels = load_split_arrays(test_df)
    print(f"      Pre-loaded all {len(df)} samples in {time.time() - t0_load:.1f}s!", flush=True)

    # 3b. Load User G-Mark Ambient Room Noise takes for negative rejection & acoustic domain augmentation
    USER_NOISE_DIR = os.path.join(BASE_DIR, "data/my_wakeword")
    noise_paths = sorted(glob.glob(os.path.join(USER_NOISE_DIR, "idle_noise_*.wav")))
    print(f"      Found {len(noise_paths)} user ambient noise recordings for acoustic adaptation.", flush=True)

    user_noise_clips = []
    for np_f in noise_paths:
        d, sr = sf.read(np_f, dtype="float32")
        if d.ndim > 1:
            d = np.mean(d, axis=1)
        if sr != TARGET_SR:
            num_samples = int(len(d) * TARGET_SR / sr)
            orig_idx = np.arange(len(d))
            target_idx = np.linspace(0, len(d) - 1, num_samples)
            d = np.interp(target_idx, orig_idx, d).astype(np.float32)
        if len(d) < TARGET_SAMPLES:
            repeats = int(np.ceil(TARGET_SAMPLES / len(d)))
            d = np.tile(d, repeats)[:TARGET_SAMPLES]
        else:
            d = d[:TARGET_SAMPLES]
        user_noise_clips.append(d)

    user_noise_np = np.array(user_noise_clips, dtype=np.float32)

    # Add ambient room noise to train set as Class 0 (_BACKGROUND_)
    # 25 copies each = 375 negative samples of user room noise
    user_bg_train = []
    for n in user_noise_np:
        for _ in range(25):
            user_bg_train.append(n)
    user_bg_train = np.array(user_bg_train, dtype=np.float32)
    user_bg_labels = np.zeros(len(user_bg_train), dtype=np.int64)

    train_wavs = np.concatenate([train_wavs, user_bg_train], axis=0)
    train_labels = np.concatenate([train_labels, user_bg_labels], axis=0)
    print(f"      Appended {len(user_bg_train)} ambient noise samples to train split as _BACKGROUND_.", flush=True)

    # GPU Ambient Noise Tensor for on-the-fly additive augmentation
    gpu_ambient_noise = torch.from_numpy(user_noise_np).unsqueeze(1).to(device)

    # Transfer to PyTorch tensors on GPU
    train_x = torch.from_numpy(train_wavs).unsqueeze(1).to(device)
    train_y = torch.tensor(train_labels, dtype=torch.long).to(device)
    val_x = torch.from_numpy(val_wavs).unsqueeze(1).to(device)
    val_y = torch.tensor(val_labels, dtype=torch.long).to(device)
    test_x = torch.from_numpy(test_wavs).unsqueeze(1).to(device)
    test_y = torch.tensor(test_labels, dtype=torch.long).to(device)

    # 4. MelSpectrogram and Data Augmentations directly on GPU
    print("[3/6] Initializing GPU Audio Pipeline...", flush=True)
    gpu_melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR, n_fft=400, win_length=400, hop_length=160, n_mels=40
    ).to(device)
    gpu_a2db = torchaudio.transforms.AmplitudeToDB().to(device)
    gpu_tmask = torchaudio.transforms.TimeMasking(20).to(device)
    gpu_fmask = torchaudio.transforms.FrequencyMasking(8).to(device)

    def transform_batch(wav_batch, is_train=True):
        if is_train:
            # 1. Random Gain (0.6x to 1.3x)
            # 1. Additive ambient noise injection (50% probability, random scale 0.2 to 0.8)
            if random.random() < 0.5 and len(gpu_ambient_noise) > 0:
                noise_idx = torch.randint(0, len(gpu_ambient_noise), (len(wav_batch),), device=device)
                scale = torch.empty(len(wav_batch), 1, 1, device=device).uniform_(0.2, 0.8)
                wav_batch = wav_batch + gpu_ambient_noise[noise_idx] * scale

            # 2. Random Gain (0.6x to 1.3x)
            gain = torch.empty(len(wav_batch), 1, 1, device=device).uniform_(0.6, 1.3)
            wav_batch = wav_batch * gain
            # 2. Time Jitter (+/- 100ms = 1600 samples)
            # 3. Time Jitter (+/- 100ms = 1600 samples)
            shift = random.randint(-1600, 1600)
            if shift > 0:
                wav_batch = torch.nn.functional.pad(wav_batch, (shift, 0))[:, :, :-shift]
            elif shift < 0:
                wav_batch = torch.nn.functional.pad(wav_batch, (0, -shift))[:, :, -shift:]

        mel = gpu_melspec(wav_batch)
        db = gpu_a2db(mel)
        if is_train:
            db = gpu_tmask(gpu_fmask(db))
        return db

    # Pre-extract Val features once for ultra-fast validation
    with torch.no_grad():
        val_feats = []
        for i in range(0, len(val_x), 128):
            val_feats.append(transform_batch(val_x[i : i + 128], is_train=False))
        val_feat_tensor = torch.cat(val_feats, dim=0)

    # 5. Model, Loss, Optimizer
    print("[4/6] Training BC-ResNet (32 Classes) on GPU...", flush=True)
    model = get_bcresnet(num_classes=num_classes).to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"      Model Parameters: {param_count:,} ({param_count * 4 / 1024:.1f} KB in FP32)", flush=True)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25, eta_min=1e-5)

    batch_size = 128
    n_train = len(train_x)
    best_val_f1 = 0.0
    best_weights = None
    t0_train = time.time()

    for epoch in range(1, 26):
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
            vout = model(val_feat_tensor)
            v_loss = criterion(vout, val_y).item()
            v_preds = vout.argmax(1).cpu().numpy()
            v_acc = accuracy_score(val_y.cpu().numpy(), v_preds)
            v_f1 = f1_score(val_y.cpu().numpy(), v_preds, average="macro", zero_division=0)

        is_best = v_f1 > best_val_f1
        if is_best:
            best_val_f1 = v_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if epoch % 5 == 0 or epoch == 25:
            print(f"      Epoch [{epoch:02d}/25] | Train Acc: {correct/n_train*100:5.2f}% | Val Acc: {v_acc*100:5.2f}% | Val Macro F1: {v_f1:.4f} {'[*BEST*]' if is_best else ''}", flush=True)

    print(f"      Training complete in {time.time() - t0_train:.1f}s! Best Val Macro F1: {best_val_f1:.4f}", flush=True)

    # 6. Final Evaluation on Unseen Test Split
    print("\n[5/6] Final Evaluation on Unseen Speaker-Disjoint Test Set (10 Speakers)...", flush=True)
    model.load_state_dict(best_weights)
    model.eval()

    with torch.no_grad():
        test_feats = []
        for i in range(0, len(test_x), 128):
            test_feats.append(transform_batch(test_x[i : i + 128], is_train=False))
        test_feat_tensor = torch.cat(test_feats, dim=0)

        tout = model(test_feat_tensor)
        t_preds = tout.argmax(1).cpu().numpy()
        t_y = test_y.cpu().numpy()

    test_acc = accuracy_score(t_y, t_preds)
    test_f1 = f1_score(t_y, t_preds, average="macro", zero_division=0)
    bg_mask = (t_y == 0)
    bg_rej = accuracy_score(t_y[bg_mask], t_preds[bg_mask]) if bg_mask.sum() > 0 else 1.0

    print("-" * 65, flush=True)
    print(f"🎯 TEST TOP-1 ACCURACY        : {test_acc*100:6.2f}%", flush=True)
    print(f"🎯 TEST MACRO F1 SCORE        : {test_f1:6.4f}", flush=True)
    print(f"🎯 BACKGROUND REJECTION RATE  : {bg_rej*100:6.2f}% ({bg_mask.sum()} clips)", flush=True)
    print("-" * 65, flush=True)

    # 7. Export ONNX & Quantize to INT8
    print("\n[6/6] Exporting ONNX FP32 and Quantizing to INT8...", flush=True)
    model.eval().to("cpu")
    dummy = torch.randn(1, 1, 40, 201)
    fp32_path = os.path.join(EXPORTS_DIR, "bcresnet_32_fp32.onnx")
    int8_path = os.path.join(EXPORTS_DIR, "bcresnet_32_int8.onnx")

    torch.onnx.export(
        model, dummy, fp32_path,
        export_params=True, opset_version=17, do_constant_folding=True,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False
    )
    print(f"      FP32 ONNX saved: {fp32_path} ({os.path.getsize(fp32_path)/1024:.1f} KB)", flush=True)

    quantize_dynamic(fp32_path, int8_path, weight_type=QuantType.QInt8)
    print(f"      INT8 Quantized Model saved: {int8_path} ({os.path.getsize(int8_path)/1024:.1f} KB)", flush=True)

    # Verify with PureNumpyFeatureExtractor
    from demo_rpi5 import PureNumpyFeatureExtractor
    ext = PureNumpyFeatureExtractor(
        os.path.join(EXPORTS_DIR, "mel_filters_40.npy"),
        os.path.join(EXPORTS_DIR, "hann_window_400.npy")
    )
    sess = ort.InferenceSession(int8_path)
    inp_name = sess.get_inputs()[0].name

    # Benchmark test sample inference latency
    sample_feat = ext.extract(test_wavs[0], TARGET_SAMPLES)
    t_start = time.perf_counter()
    for _ in range(50):
        _ = sess.run(None, {inp_name: sample_feat})
    lat_ms = (time.perf_counter() - t_start) / 50 * 1000.0
    print(f"      INT8 ONNX Inference Latency (CPU single thread): {lat_ms:.2f} ms", flush=True)

    # Benchmark directly on all user ambient noise takes with PureNumpyFeatureExtractor
    print("\n      --- Verifying Rejection on User G-Mark Ambient Noise Takes (Target: _BACKGROUND_) ---", flush=True)
    noise_eval_scores = []
    for np_f in noise_paths:
        d = load_audio_sample(np_f)
        feat = ext.extract(d, TARGET_SAMPLES)
        logits = sess.run(None, {inp_name: feat})[0]
        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_l / np.sum(exp_l, axis=1, keepdims=True)
        pred_idx = int(np.argmax(probs, axis=1)[0])
        pred_label = idx2label[pred_idx]
        bg_prob = float(probs[0, 0])
        name = os.path.basename(np_f)
        status = "✅ REJECTED (PASS)" if pred_label == "_BACKGROUND_" else f"⚠️ FALSE COMMAND ({pred_label})"
        print(f"      Noise {name}: Predicted = {pred_label:<14} (Background Conf: {bg_prob*100:5.2f}%)  {status}", flush=True)
        noise_eval_scores.append(bg_prob)

    avg_bg_conf = np.mean(noise_eval_scores) * 100
    print("-" * 65, flush=True)
    print(f"🎯 AVERAGE AMBIENT NOISE REJECTION CONFIDENCE: {avg_bg_conf:.2f}%", flush=True)
    print("-" * 65, flush=True)

    # Save metrics summary
    metrics = {
        "model": "bcresnet_32",
        "num_classes": num_classes,
        "parameters": param_count,
        "test_accuracy": float(test_acc),
        "test_macro_f1": float(test_f1),
        "bg_rejection_rate": float(bg_rej),
        "fp32_size_kb": os.path.getsize(fp32_path) / 1024,
        "int8_size_kb": os.path.getsize(int8_path) / 1024,
        "cpu_latency_ms": float(lat_ms)
    }
    with open(os.path.join(EXPORTS_DIR, "test_metrics_bcresnet_32.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"      Metrics written to exports/test_metrics_bcresnet_32.json", flush=True)
    print("=" * 65, flush=True)


if __name__ == "__main__":
    main()
