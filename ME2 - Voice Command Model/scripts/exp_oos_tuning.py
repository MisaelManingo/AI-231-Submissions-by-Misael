#!/usr/bin/env python3
"""
exp_oos_tuning.py

Ablation script to evaluate oversampling of real OOS speech and higher OOS class weight.
Tuned strictly on validation set only.
Reports:
- Class-only FAR on OOS speech (validation)
- Class-only FRR on real Filipino speech (validation)
- In-scope validation accuracy
- Real Filipino validation accuracy
"""

import os
import sys
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.bcresnet import get_bcresnet

CACHE_DIR = os.path.join(BASE_DIR, "data", "v3_cache_94class")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports", "v3_94class")
TARGET_SR = 16000
NUM_CLASSES = 94
OOS_CLASS_IDX = 93

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run_experiment(oos_oversample=1, oos_weight=2.50, filipino_oversample=4, epochs=15, device="cuda:6"):
    set_seed(42)
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    print(f"\n--- Testing OOS Oversample={oos_oversample}x, OOS Weight={oos_weight:.2f}, Fil Oversample={filipino_oversample}x ---", flush=True)

    train_npz = np.load(os.path.join(CACHE_DIR, "train_data.npz"))
    val_npz = np.load(os.path.join(CACHE_DIR, "val_data.npz"))
    user_noise_np = np.load(os.path.join(CACHE_DIR, "user_ambient_noise.npy"))

    train_wavs_raw = train_npz["wavs"]
    train_labels_raw = train_npz["labels_94"]
    train_vt_raw = train_npz["voice_types"]
    train_oos_raw = train_npz["is_oos_speech"]

    fil_indices = np.where(train_vt_raw == "real_filipino")[0]
    num_fil_dups = max(filipino_oversample - 1, 0)

    oos_speech_indices = np.where(train_oos_raw == 1)[0]
    num_oos_dups = max(oos_oversample - 1, 0)

    oversampled_indices = np.concatenate([
        np.arange(len(train_labels_raw)),
        np.repeat(fil_indices, num_fil_dups),
        np.repeat(oos_speech_indices, num_oos_dups)
    ])

    train_wavs = torch.from_numpy(train_wavs_raw).unsqueeze(1).to(dev)
    train_labels = torch.from_numpy(train_labels_raw).to(dev)
    train_idx_t = torch.from_numpy(oversampled_indices).to(dev)

    val_wavs = torch.from_numpy(val_npz["wavs"]).unsqueeze(1).to(dev)
    val_labels_np = val_npz["labels_94"]
    val_fg = val_npz["is_filipino_group"]
    val_oos = val_npz["is_oos_speech"]

    ambient_noise = torch.from_numpy(user_noise_np).unsqueeze(1).to(dev)

    gpu_melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR, n_fft=400, win_length=400, hop_length=160, n_mels=40
    ).to(dev)
    gpu_a2db = torchaudio.transforms.AmplitudeToDB().to(dev)
    gpu_tmask = torchaudio.transforms.TimeMasking(20).to(dev)
    gpu_fmask = torchaudio.transforms.FrequencyMasking(6).to(dev)

    def extract_features(x, augment=False):
        if augment:
            shift = random.randint(-1600, 1600)
            if shift > 0:
                x = F.pad(x[:, :, :-shift], (shift, 0))
            elif shift < 0:
                x = F.pad(x[:, :, -shift:], (0, -shift))
            if random.random() < 0.5:
                n_idx = random.randint(0, len(ambient_noise) - 1)
                noise_clip = ambient_noise[n_idx : n_idx + 1]
                snr_db = random.uniform(10.0, 25.0)
                sig_p = torch.mean(x**2, dim=-1, keepdim=True) + 1e-8
                noi_p = torch.mean(noise_clip**2, dim=-1, keepdim=True) + 1e-8
                snr_lin = 10.0 ** (snr_db / 10.0)
                scale = torch.sqrt(sig_p / (snr_lin * noi_p))
                x = x + scale * noise_clip

        mel = gpu_melspec(x)
        log_mel = gpu_a2db(mel)
        if augment:
            log_mel = gpu_tmask(log_mel)
            log_mel = gpu_fmask(log_mel)
        return log_mel

    val_feats_list = []
    with torch.no_grad():
        for i in range(0, len(val_wavs), 256):
            val_feats_list.append(extract_features(val_wavs[i : i + 256], augment=False))
    val_feats = torch.cat(val_feats_list, dim=0)

    model = get_bcresnet(num_classes=NUM_CLASSES).to(dev)
    class_weights = torch.ones(NUM_CLASSES, device=dev)
    class_weights[OOS_CLASS_IDX] = oos_weight
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    batch_size = 64
    best_fil_acc = -1.0
    best_metrics = {}

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(train_idx_t), device=dev)
        total_loss, n_batches = 0.0, 0
        for b_start in range(0, len(train_idx_t), batch_size):
            b_idx = train_idx_t[perm[b_start : b_start + batch_size]]
            bx_wav = train_wavs[b_idx]
            by = train_labels[b_idx]
            bx = extract_features(bx_wav, augment=True)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        scheduler.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(val_feats)
            v_preds = torch.argmax(v_logits, dim=1).cpu().numpy()

        in_scope_mask = (val_labels_np != OOS_CLASS_IDX)
        acc_in_scope = np.mean(v_preds[in_scope_mask] == val_labels_np[in_scope_mask]) * 100.0

        fil_mask = (val_fg == 1) & in_scope_mask
        acc_fil = np.mean(v_preds[fil_mask] == val_labels_np[fil_mask]) * 100.0 if np.sum(fil_mask) > 0 else 0.0

        oos_speech_mask = (val_oos == 1)
        oos_accepted = np.sum(oos_speech_mask & (v_preds != OOS_CLASS_IDX))
        n_oos = np.sum(oos_speech_mask)
        far_class_only = (oos_accepted / n_oos) * 100.0 if n_oos > 0 else 0.0

        fil_rejected = np.sum(fil_mask & (v_preds == OOS_CLASS_IDX))
        n_fil = np.sum(fil_mask)
        frr_fil_class_only = (fil_rejected / n_fil) * 100.0 if n_fil > 0 else 0.0

        if acc_fil >= best_fil_acc:
            best_fil_acc = acc_fil
            best_metrics = {
                "epoch": epoch,
                "val_in_scope_acc": round(float(acc_in_scope), 2),
                "val_filipino_acc": round(float(acc_fil), 2),
                "class_only_far_oos_speech": round(float(far_class_only), 2),
                "class_only_frr_filipino": round(float(frr_fil_class_only), 2),
            }
        print(f"  Epoch {epoch:2d}/{epochs} | Val In-Scope Acc: {acc_in_scope:5.2f}% | Val Fil Acc: {acc_fil:5.2f}% | Class-only FAR: {far_class_only:5.2f}% | FRR: {frr_fil_class_only:5.2f}%", flush=True)

    print(f"  --> Best Epoch {best_metrics['epoch']}: InScope={best_metrics['val_in_scope_acc']}%, FilAcc={best_metrics['val_filipino_acc']}%, ClassOnlyFAR={best_metrics['class_only_far_oos_speech']}%, ClassOnlyFRR={best_metrics['class_only_frr_filipino']}%")
    return best_metrics

if __name__ == "__main__":
    configs = [
        {"oos_oversample": 1, "oos_weight": 2.50},
        {"oos_oversample": 3, "oos_weight": 4.00},
        {"oos_oversample": 4, "oos_weight": 5.00},
        {"oos_oversample": 5, "oos_weight": 6.00},
    ]
    results = []
    for c in configs:
        res = run_experiment(oos_oversample=c["oos_oversample"], oos_weight=c["oos_weight"], epochs=15, device="cuda:6")
        res.update(c)
        results.append(res)

    print("\n" + "=" * 80)
    print("OOS LEARNING ABLATION RESULTS (VALIDATION SET ONLY)")
    print("=" * 80)
    print(f"{'Config':<20} | {'Class-Only FAR':<16} | {'Class-Only FRR':<16} | {'Filipino Acc':<14} | {'In-Scope Acc':<14}")
    print("-" * 80)
    for r in results:
        cfg_name = f"OOS {r['oos_oversample']}x, wt={r['oos_weight']:.1f}"
        print(f"{cfg_name:<20} | {r['class_only_far_oos_speech']:>14.2f}% | {r['class_only_frr_filipino']:>14.2f}% | {r['val_filipino_acc']:>12.2f}% | {r['val_in_scope_acc']:>12.2f}%")
    print("=" * 80)
