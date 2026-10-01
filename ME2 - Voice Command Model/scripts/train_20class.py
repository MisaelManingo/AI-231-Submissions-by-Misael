#!/usr/bin/env python3
"""
train_20class.py

Trains 20-class Voice Command Models (BC-ResNet-1 and baseline DS-CNN) on the A100 GPU:
- 19 schema commands + OUT_OF_SCOPE as the 20th class (index 19).
- Solves Filipino speech starvation via 4x oversampling of real Filipino clips during training.
- Class-weighted loss with logged effective weight for OUT_OF_SCOPE.
- Multi-seed training (seeds 42, 1337, 2026).
- Strict validation-only checkpoint selection.
- Dual-constraint threshold tau tuning on validation only:
    Constraint 1: FAR on OOS speech <= 5.0%
    Constraint 2: FRR on real Filipino speech <= 30.0%
- Temperature scaling calibration evaluated on validation logits.
- Evaluates rejection rules: class-only, threshold-only, combined, and margin variant.
- Computes FAR on OOS speech (with Wilson 95% CI), FAR on mic noise, accuracy on accepted,
  and False-Reject Rate (FRR) on real Filipino group recordings.
- Test accuracy broken down by voice type: real Filipino, open-source, synthetic.
- Leaves holdout evaluation for physical Raspberry Pi execution (default --eval-holdout False).
"""

import os
import sys
import time
import json
import socket
import argparse
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.bcresnet import get_bcresnet
from models.dscnn import get_dscnn

CACHE_DIR = os.path.join(BASE_DIR, "data", "v2_cache_20class")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports", "v2_20class")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "v2")
os.makedirs(EXPORTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

TARGET_SR = 16000
TARGET_SAMPLES = 32000
OOS_CLASS_IDX = 19
OOS_EFFECTIVE_WEIGHT = 2.50  # Up-weighting OUT_OF_SCOPE for vigilant negative rejection
FILIPINO_OVERSAMPLE_FACTOR = 4  # 4x oversampling of real Filipino speech to fix starvation


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def wilson_ci(k, n, confidence=0.95):
    """Computes Wilson score interval for binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z = 1.95996
    denom = 1 + (z**2) / n
    centre = (p + (z**2) / (2 * n)) / denom
    margin = (z / denom) * np.sqrt((p * (1 - p)) / n + (z**2) / (4 * (n**2)))
    lower = max(0.0, centre - margin)
    upper = min(1.0, centre + margin)
    return round(p * 100, 2), round(lower * 100, 2), round(upper * 100, 2)


def fit_temperature_scaling(val_logits_t, val_labels_t):
    """
    Fits a single scalar temperature T > 0 on validation logits using L-BFGS
    to minimize cross-entropy loss. Returns optimal temperature T.
    """
    temperature = nn.Parameter(torch.ones(1, device=val_logits_t.device) * 1.5)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.LBFGS([temperature], lr=0.05, max_iter=50)

    def eval_closure():
        optimizer.zero_grad()
        loss = criterion(val_logits_t / temperature, val_labels_t)
        loss.backward()
        return loss

    optimizer.step(eval_closure)
    t_val = float(temperature.item())
    return max(round(t_val, 4), 0.05)


def evaluate_rules_on_split(probs, labels, is_filipino_group, is_oos_speech, tau, delta_m, mic_noise_probs=None):
    """
    Evaluates 4 rejection rules on a dataset split:
    1. Class-only rule: reject if argmax == 19
    2. Threshold-only rule: reject if max_prob < tau
    3. Combined rule: reject if argmax == 19 OR max_prob < tau
    4. Margin variant: reject if argmax == 19 OR (top1_prob - top2_prob) < delta_m
    """
    preds = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)

    sorted_probs = np.sort(probs, axis=1)
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2]
    margins = top1 - top2

    in_scope_mask = (labels != OOS_CLASS_IDX)
    oos_speech_mask = (is_oos_speech == 1)
    filipino_in_scope_mask = (is_filipino_group == 1)

    n_oos = int(np.sum(oos_speech_mask))
    n_filipino = int(np.sum(filipino_in_scope_mask))

    rule_definitions = {
        "class_only": (preds == OOS_CLASS_IDX),
        "threshold_only": (max_probs < tau),
        "combined": (preds == OOS_CLASS_IDX) | (max_probs < tau),
        "margin_variant": (preds == OOS_CLASS_IDX) | (margins < delta_m),
    }

    results = {}
    for rule_name, reject_mask in rule_definitions.items():
        accept_mask = ~reject_mask

        # 1. FAR on OOS speech
        oos_accepted = np.sum(oos_speech_mask & accept_mask)
        far_val, far_low, far_high = wilson_ci(oos_accepted, n_oos)

        # 2. FAR on physical mic-noise clips
        if mic_noise_probs is not None and len(mic_noise_probs) > 0:
            noise_preds = np.argmax(mic_noise_probs, axis=1)
            noise_max_p = np.max(mic_noise_probs, axis=1)
            noise_sorted = np.sort(mic_noise_probs, axis=1)
            noise_m = noise_sorted[:, -1] - noise_sorted[:, -2]

            if rule_name == "class_only":
                n_reject = (noise_preds == OOS_CLASS_IDX)
            elif rule_name == "threshold_only":
                n_reject = (noise_max_p < tau)
            elif rule_name == "combined":
                n_reject = (noise_preds == OOS_CLASS_IDX) | (noise_max_p < tau)
            else:
                n_reject = (noise_preds == OOS_CLASS_IDX) | (noise_m < delta_m)

            noise_accepted = int(np.sum(~n_reject))
            far_noise = round((noise_accepted / len(mic_noise_probs)) * 100.0, 2)
        else:
            far_noise = 0.0

        # 3. Command accuracy on accepted in-scope clips
        accepted_in_scope = in_scope_mask & accept_mask
        if np.sum(accepted_in_scope) > 0:
            correct_accepted = np.sum(accepted_in_scope & (preds == labels))
            acc_accepted = round((correct_accepted / np.sum(accepted_in_scope)) * 100.0, 2)
        else:
            acc_accepted = 0.0

        # 4. False-Reject Rate (FRR) on real Filipino group recordings
        if n_filipino > 0:
            filipino_rejected = np.sum(filipino_in_scope_mask & reject_mask)
            frr_filipino = round((filipino_rejected / n_filipino) * 100.0, 2)
        else:
            frr_filipino = 0.0

        results[rule_name] = {
            "far_oos_speech_pct": far_val,
            "far_oos_ci_95": [far_low, far_high],
            "far_mic_noise_pct": far_noise,
            "command_acc_accepted_pct": acc_accepted,
            "frr_filipino_group_pct": frr_filipino,
            "accepted_total": int(np.sum(accept_mask)),
            "rejected_total": int(np.sum(reject_mask)),
        }

    return results


def run_tau_sweep_on_validation(val_probs, val_labels, is_filipino_group, is_oos_speech, target_far=5.0, max_frr_cap=30.0):
    """
    Sweeps tau on validation set from 0.10 to 0.95 in 0.05 steps.
    Dual-constraint selection:
      - Constraint 1: FAR on OOS speech <= target_far (5.0%)
      - Constraint 2: FRR on real Filipino speech <= max_frr_cap (30.0%)
    """
    sweep_rows = []
    best_tau = 0.50
    best_frr = float("inf")

    tau_candidates = [round(t, 2) for t in np.arange(0.10, 0.96, 0.05)]

    for tau in tau_candidates:
        res = evaluate_rules_on_split(
            probs=val_probs,
            labels=val_labels,
            is_filipino_group=is_filipino_group,
            is_oos_speech=is_oos_speech,
            tau=tau,
            delta_m=0.15,
            mic_noise_probs=None,
        )
        comb = res["combined"]
        far = comb["far_oos_speech_pct"]
        frr = comb["frr_filipino_group_pct"]
        acc = comb["command_acc_accepted_pct"]

        satisfies_both = (far <= target_far) and (frr <= max_frr_cap)
        sweep_rows.append({
            "tau": tau,
            "val_far_oos_pct": far,
            "val_frr_filipino_pct": frr,
            "val_acc_accepted_pct": acc,
            "satisfies_dual_constraints": satisfies_both,
        })

        if satisfies_both:
            # When both constraints are satisfied, pick tau that minimizes FRR
            if frr < best_frr or best_frr == float("inf"):
                best_frr = frr
                best_tau = tau

    # Fallback if no tau strictly satisfies both
    if best_frr == float("inf"):
        best_penalty = float("inf")
        for row in sweep_rows:
            penalty = max(row["val_far_oos_pct"] - target_far, 0.0) * 2.0 + max(row["val_frr_filipino_pct"] - max_frr_cap, 0.0)
            if penalty < best_penalty:
                best_penalty = penalty
                best_tau = row["tau"]

    df_sweep = pd.DataFrame(sweep_rows)
    return best_tau, df_sweep


def compute_voice_type_breakdown(probs, labels, voice_types, tau, delta_m):
    """Computes test accuracy and acceptance metrics broken down by voice type."""
    preds = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)
    combined_reject = (preds == OOS_CLASS_IDX) | (max_probs < tau)
    combined_accept = ~combined_reject

    v_types = ["real_filipino", "open_source", "synthetic"]
    vt_results = {}

    for vt in v_types:
        vt_mask = (voice_types == vt)
        n_vt = int(np.sum(vt_mask))
        if n_vt == 0:
            continue

        vt_labels = labels[vt_mask]
        vt_preds = preds[vt_mask]
        raw_acc = round(float(np.mean(vt_preds == vt_labels) * 100.0), 2)
        raw_f1 = round(float(f1_score(vt_labels, vt_preds, average="macro", zero_division=0)), 4)

        # On accepted in-scope clips
        in_scope_mask = (labels != OOS_CLASS_IDX)
        vt_accepted_in_scope = vt_mask & in_scope_mask & combined_accept
        n_acc_in_scope = int(np.sum(vt_accepted_in_scope))

        if n_acc_in_scope > 0:
            acc_on_acc = round(float(np.sum(vt_accepted_in_scope & (preds == labels)) / n_acc_in_scope * 100.0), 2)
        else:
            acc_on_acc = 0.0

        # False reject rate on in-scope clips of this voice type
        vt_in_scope = vt_mask & in_scope_mask
        n_in_scope = int(np.sum(vt_in_scope))
        if n_in_scope > 0:
            vt_frr = round(float(np.sum(vt_in_scope & combined_reject) / n_in_scope * 100.0), 2)
        else:
            vt_frr = 0.0

        vt_results[vt] = {
            "total_clips": n_vt,
            "in_scope_clips": n_in_scope,
            "raw_accuracy_pct": raw_acc,
            "raw_macro_f1": raw_f1,
            "accepted_in_scope_count": n_acc_in_scope,
            "accuracy_on_accepted_pct": acc_on_acc,
            "frr_pct": vt_frr,
        }

    return vt_results


def train_single_run(model_name, seed, device, epochs, batch_size, lr, eval_holdout=False):
    set_seed(seed)
    print(f"\n--- Training {model_name.upper()} (20 classes) | Seed: {seed} ---", flush=True)

    # 1. Load Preprocessed 20-Class Data
    train_npz = np.load(os.path.join(CACHE_DIR, "train_data.npz"))
    val_npz = np.load(os.path.join(CACHE_DIR, "val_data.npz"))
    test_npz = np.load(os.path.join(CACHE_DIR, "test_data.npz"))
    user_noise_np = np.load(os.path.join(CACHE_DIR, "user_ambient_noise.npy"))

    train_wavs_raw = train_npz["wavs"]
    train_labels_raw = train_npz["labels"]
    train_vt_raw = train_npz["voice_types"]

    # Filipino speech oversampling (4x: 1 original + 3 duplicates)
    fil_mask = (train_vt_raw == "real_filipino")
    fil_indices = np.where(fil_mask)[0]
    num_duplicates = FILIPINO_OVERSAMPLE_FACTOR - 1
    oversampled_train_indices = np.concatenate([
        np.arange(len(train_labels_raw)),
        np.repeat(fil_indices, num_duplicates)
    ])

    train_wavs = torch.from_numpy(train_wavs_raw).unsqueeze(1).to(device)
    train_labels = torch.from_numpy(train_labels_raw).to(device)
    train_idx_t = torch.from_numpy(oversampled_train_indices).to(device)

    total_train_samples = len(oversampled_train_indices)
    total_fil_samples = len(fil_indices) * FILIPINO_OVERSAMPLE_FACTOR
    fil_share_pct = round((total_fil_samples / total_train_samples) * 100.0, 2)
    print(f"  [Filipino Oversampling] Factor: {FILIPINO_OVERSAMPLE_FACTOR}x | Unique: {len(fil_indices)} | Per epoch: {total_fil_samples}/{total_train_samples} ({fil_share_pct}% effective share)")

    val_wavs = torch.from_numpy(val_npz["wavs"]).unsqueeze(1).to(device)
    val_labels = torch.from_numpy(val_npz["labels"]).to(device)
    val_labels_np = val_npz["labels"]
    val_fg = val_npz["is_filipino_group"]
    val_oos = val_npz["is_oos_speech"]
    val_vt = val_npz["voice_types"]

    test_wavs = torch.from_numpy(test_npz["wavs"]).unsqueeze(1).to(device)
    test_labels_np = test_npz["labels"]
    test_fg = test_npz["is_filipino_group"]
    test_oos = test_npz["is_oos_speech"]
    test_vt = test_npz["voice_types"]

    ambient_noise = torch.from_numpy(user_noise_np).unsqueeze(1).to(device)

    # 2. Audio Front-End & Augmentations
    gpu_melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR, n_fft=400, win_length=400, hop_length=160, n_mels=40
    ).to(device)
    gpu_a2db = torchaudio.transforms.AmplitudeToDB().to(device)
    gpu_tmask = torchaudio.transforms.TimeMasking(20).to(device)
    gpu_fmask = torchaudio.transforms.FrequencyMasking(6).to(device)

    def extract_features(x, augment=False):
        if augment:
            if random.random() < 0.6 and len(ambient_noise) > 0:
                n_idx = torch.randint(0, len(ambient_noise), (x.size(0),), device=device)
                beta = (torch.rand(x.size(0), 1, 1, device=device) * 0.6) + 0.2
                x = x + beta * ambient_noise[n_idx]
            if random.random() < 0.5:
                shift = random.randint(-1600, 1600)
                x = torch.roll(x, shifts=shift, dims=-1)

        mel = gpu_melspec(x)
        db = gpu_a2db(mel)
        if augment:
            db = gpu_tmask(db)
            db = gpu_fmask(db)
        mean = db.mean(dim=(-2, -1), keepdim=True)
        std = db.std(dim=(-2, -1), keepdim=True) + 1e-5
        return (db - mean) / std

    # 3. Model Architecture & Class Weighting
    num_classes = 20
    if model_name == "bcresnet":
        model = get_bcresnet(num_classes=num_classes).to(device)
    elif model_name == "dscnn":
        model = get_dscnn(num_classes=num_classes).to(device)
    else:
        raise ValueError(f"Unknown model {model_name}")

    class_weights = torch.ones(num_classes, dtype=torch.float32, device=device)
    class_weights[OOS_CLASS_IDX] = OOS_EFFECTIVE_WEIGHT
    print(f"  Class weights initialized. OUT_OF_SCOPE effective weight: {OOS_EFFECTIVE_WEIGHT}")

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_acc = -1.0
    best_ckpt_path = os.path.join(CKPT_DIR, f"{model_name}_20class_seed{seed}_best.pt")
    log_rows = []

    t0 = time.time()
    n_oversampled = len(train_idx_t)

    for epoch in range(1, epochs + 1):
        model.train()
        perm = train_idx_t[torch.randperm(n_oversampled, device=device)]
        running_loss = 0.0
        correct = 0
        total = 0

        for b_start in range(0, n_oversampled, batch_size):
            b_idx = perm[b_start : b_start + batch_size]
            bx = train_wavs[b_idx]
            by = train_labels[b_idx]

            feats = extract_features(bx, augment=True)
            optimizer.zero_grad()
            logits = model(feats)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * len(by)
            preds = logits.argmax(dim=1)
            correct += (preds == by).sum().item()
            total += len(by)

        scheduler.step()
        train_loss = running_loss / total
        train_acc = (correct / total) * 100.0

        # Evaluate on validation split
        model.eval()
        with torch.no_grad():
            v_correct = 0
            v_total = 0
            v_loss = 0.0
            for b_start in range(0, len(val_labels), batch_size):
                bx = val_wavs[b_start : b_start + batch_size]
                by = val_labels[b_start : b_start + batch_size]
                feats = extract_features(bx, augment=False)
                logits = model(feats)
                loss = criterion(logits, by)
                v_loss += loss.item() * len(by)
                preds = logits.argmax(dim=1)
                v_correct += (preds == by).sum().item()
                v_total += len(by)

            val_loss = v_loss / v_total
            val_acc = (v_correct / v_total) * 100.0

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "val_acc": val_acc,
                    "seed": seed,
                    "model_name": model_name,
                    "oos_effective_weight": OOS_EFFECTIVE_WEIGHT,
                    "filipino_oversample_factor": FILIPINO_OVERSAMPLE_FACTOR,
                    "filipino_effective_share_pct": fil_share_pct,
                },
                best_ckpt_path,
            )

        log_rows.append({
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "train_acc": round(train_acc, 2),
            "val_loss": round(val_loss, 4),
            "val_acc": round(val_acc, 2),
            "is_best": is_best,
        })

        if epoch % 5 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:2d}/{epochs:2d} | Train: Loss {train_loss:.4f}, Acc {train_acc:5.2f}% | Val: Loss {val_loss:.4f}, Acc {val_acc:5.2f}% {'[BEST]' if is_best else ''}", flush=True)

    wall_clock = time.time() - t0

    # Save CSV Log
    csv_path = os.path.join(EXPORTS_DIR, f"training_log_{model_name}_20class_seed{seed}.csv")
    pd.DataFrame(log_rows).to_csv(csv_path, index=False)

    # 4. Load Best Checkpoint and Run Predictions
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    def get_logits_tensor(wav_t):
        logits_all = []
        with torch.no_grad():
            for b_start in range(0, len(wav_t), batch_size):
                bx = wav_t[b_start : b_start + batch_size]
                feats = extract_features(bx, augment=False)
                logits = model(feats)
                logits_all.append(logits)
        return torch.cat(logits_all, dim=0)

    val_logits_t = get_logits_tensor(val_wavs)
    test_logits_t = get_logits_tensor(test_wavs)

    val_probs_uncal = F.softmax(val_logits_t, dim=1).cpu().numpy()
    test_probs_uncal = F.softmax(test_logits_t, dim=1).cpu().numpy()

    # Temperature Scaling Evaluation
    temp_opt = fit_temperature_scaling(val_logits_t, val_labels)
    val_probs_cal = F.softmax(val_logits_t / temp_opt, dim=1).cpu().numpy()
    test_probs_cal = F.softmax(test_logits_t / temp_opt, dim=1).cpu().numpy()
    print(f"  --> Temperature Scaling: Fitted optimal T = {temp_opt:.4f}")

    # Predict physical mic-noise clips
    mic_noise_wavs = torch.from_numpy(user_noise_np).unsqueeze(1).to(device)
    mic_noise_logits_t = get_logits_tensor(mic_noise_wavs)
    mic_noise_probs = F.softmax(mic_noise_logits_t, dim=1).cpu().numpy()

    # 5. Dual-Constraint Threshold Tuning on Validation Only (Uncalibrated)
    best_tau, df_sweep = run_tau_sweep_on_validation(
        val_probs=val_probs_uncal,
        val_labels=val_labels_np,
        is_filipino_group=val_fg,
        is_oos_speech=val_oos,
        target_far=5.0,
        max_frr_cap=30.0,
    )
    print(f"  --> Dual-Constraint Tuned tau* = {best_tau:.2f} (Target FAR <= 5.0%, Max Filipino FRR <= 30.0%)")

    # Sweep on calibrated probabilities for comparison
    best_tau_cal, df_sweep_cal = run_tau_sweep_on_validation(
        val_probs=val_probs_cal,
        val_labels=val_labels_np,
        is_filipino_group=val_fg,
        is_oos_speech=val_oos,
        target_far=5.0,
        max_frr_cap=30.0,
    )
    print(f"  --> Temperature-Calibrated Tuned tau* = {best_tau_cal:.2f}")

    # Save validation tau sweep tables
    sweep_path = os.path.join(EXPORTS_DIR, f"tau_sweep_val_{model_name}_seed{seed}.csv")
    df_sweep.to_csv(sweep_path, index=False)

    sweep_cal_path = os.path.join(EXPORTS_DIR, f"tau_sweep_val_{model_name}_calibrated_seed{seed}.csv")
    df_sweep_cal.to_csv(sweep_cal_path, index=False)

    # 6. Evaluate Rejection Rules on Unseen Test Split
    test_rule_results = evaluate_rules_on_split(
        probs=test_probs_uncal,
        labels=test_labels_np,
        is_filipino_group=test_fg,
        is_oos_speech=test_oos,
        tau=best_tau,
        delta_m=0.15,
        mic_noise_probs=mic_noise_probs,
    )

    # Voice Type Breakdown on Test Split
    vt_breakdown = compute_voice_type_breakdown(
        probs=test_probs_uncal,
        labels=test_labels_np,
        voice_types=test_vt,
        tau=best_tau,
        delta_m=0.15,
    )

    # Raw overall test metrics
    test_preds = np.argmax(test_probs_uncal, axis=1)
    test_kw_acc = round(float(np.mean(test_preds == test_labels_np) * 100.0), 2)
    test_f1 = round(float(f1_score(test_labels_np, test_preds, average="macro", zero_division=0)), 4)

    comb_rule = test_rule_results["combined"]
    print(f"  --> Seed {seed} Test Results (tau={best_tau:.2f}):")
    print(f"      Overall Test Accuracy: {test_kw_acc}% | Macro F1: {test_f1}")
    print(f"      [Combined Rule]: FAR OOS={comb_rule['far_oos_speech_pct']}% (95% CI: {comb_rule['far_oos_ci_95']}), FAR Mic Noise={comb_rule['far_mic_noise_pct']}%, Acc Accepted={comb_rule['command_acc_accepted_pct']}% (FRR Filipino: {comb_rule['frr_filipino_group_pct']}%)")
    print(f"      [Voice Type Breakdown Acc]: Real Filipino={vt_breakdown['real_filipino']['raw_accuracy_pct']}%, Open Source={vt_breakdown['open_source']['raw_accuracy_pct']}%, Synthetic={vt_breakdown['synthetic']['raw_accuracy_pct']}%")

    return {
        "seed": seed,
        "best_epoch": ckpt["epoch"],
        "val_acc": round(float(ckpt["val_acc"]), 2),
        "wall_clock_time_s": round(wall_clock, 2),
        "best_tau": best_tau,
        "temperature_optimal": temp_opt,
        "test_overall": {
            "keyword_accuracy": test_kw_acc,
            "macro_f1": test_f1,
        },
        "test_rules": test_rule_results,
        "voice_type_breakdown": vt_breakdown,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="both", choices=["bcresnet", "dscnn", "both"])
    parser.add_argument("--seeds", type=str, default="42,1337,2026")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--gpu", type=int, default=7)
    parser.add_argument("--eval-holdout", action="store_true", default=False, help="Holdout evaluation (default: False on cluster)")
    args = parser.parse_args()

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    node_id = socket.gethostname()
    gpu_name = torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else "CPU"
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    print("=" * 70)
    print("🚀 20-CLASS MULTI-SEED VCM TRAINING (A100 CLUSTER - BALANCED FILIPINO)")
    print("=" * 70)
    print(f"  Node ID:           {node_id}")
    print(f"  Device:            {device} ({gpu_name})")
    print(f"  Seeds:             {seeds}")
    print(f"  OOS Class Weight:  {OOS_EFFECTIVE_WEIGHT}")
    print(f"  Filipino Oversample:{FILIPINO_OVERSAMPLE_FACTOR}x")
    print(f"  Dual Constraints:  Target FAR <= 5.0% & Max Filipino FRR <= 30.0%")
    print(f"  Eval Holdout:      {args.eval_holdout} (Cluster policy: False)")

    models_to_run = ["bcresnet", "dscnn"] if args.model == "both" else [args.model]
    all_summaries = {}

    for m_name in models_to_run:
        print("\n" + "#" * 70)
        print(f"🏁 RUNNING MULTI-SEED EXPERIMENTS: {m_name.upper()}")
        print("#" * 70)

        seed_results = []
        for s in seeds:
            res = train_single_run(
                model_name=m_name,
                seed=s,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                eval_holdout=args.eval_holdout,
            )
            seed_results.append(res)

        # Aggregate across seeds
        accs = [r["test_overall"]["keyword_accuracy"] for r in seed_results]
        f1s = [r["test_overall"]["macro_f1"] for r in seed_results]
        best_taus = [r["best_tau"] for r in seed_results]
        temps = [r["temperature_optimal"] for r in seed_results]

        comb_far_oos = [r["test_rules"]["combined"]["far_oos_speech_pct"] for r in seed_results]
        comb_far_noise = [r["test_rules"]["combined"]["far_mic_noise_pct"] for r in seed_results]
        comb_acc_acc = [r["test_rules"]["combined"]["command_acc_accepted_pct"] for r in seed_results]
        comb_frr_fil = [r["test_rules"]["combined"]["frr_filipino_group_pct"] for r in seed_results]

        # Voice type aggregations
        fil_accs = [r["voice_type_breakdown"]["real_filipino"]["raw_accuracy_pct"] for r in seed_results]
        os_accs = [r["voice_type_breakdown"]["open_source"]["raw_accuracy_pct"] for r in seed_results]
        syn_accs = [r["voice_type_breakdown"]["synthetic"]["raw_accuracy_pct"] for r in seed_results]

        fil_acc_acc = [r["voice_type_breakdown"]["real_filipino"]["accuracy_on_accepted_pct"] for r in seed_results]
        fil_frrs = [r["voice_type_breakdown"]["real_filipino"]["frr_pct"] for r in seed_results]

        summary = {
            "model_name": m_name,
            "architecture": "BC-ResNet-1" if m_name == "bcresnet" else "DS-CNN",
            "cluster": {
                "node_id": node_id,
                "gpu_name": gpu_name,
                "gpu_count": 1,
            },
            "oos_effective_weight": OOS_EFFECTIVE_WEIGHT,
            "filipino_oversample_factor": FILIPINO_OVERSAMPLE_FACTOR,
            "seeds": seeds,
            "per_seed_results": seed_results,
            "aggregate_test_cluster": {
                "accuracy_mean": round(float(np.mean(accs)), 2),
                "accuracy_std": round(float(np.std(accs)), 2),
                "macro_f1_mean": round(float(np.mean(f1s)), 4),
                "macro_f1_std": round(float(np.std(f1s)), 4),
                "best_tau_mean": round(float(np.mean(best_taus)), 2),
                "temperature_mean": round(float(np.mean(temps)), 4),
                "combined_rule": {
                    "far_oos_speech_mean": round(float(np.mean(comb_far_oos)), 2),
                    "far_oos_speech_std": round(float(np.std(comb_far_oos)), 2),
                    "far_mic_noise_mean": round(float(np.mean(comb_far_noise)), 2),
                    "command_acc_accepted_mean": round(float(np.mean(comb_acc_acc)), 2),
                    "command_acc_accepted_std": round(float(np.std(comb_acc_acc)), 2),
                    "frr_filipino_group_mean": round(float(np.mean(comb_frr_fil)), 2),
                    "frr_filipino_group_std": round(float(np.std(comb_frr_fil)), 2),
                },
                "voice_type_breakdown": {
                    "real_filipino_acc_mean": round(float(np.mean(fil_accs)), 2),
                    "real_filipino_acc_std": round(float(np.std(fil_accs)), 2),
                    "real_filipino_acc_on_accepted_mean": round(float(np.mean(fil_acc_acc)), 2),
                    "real_filipino_frr_mean": round(float(np.mean(fil_frrs)), 2),
                    "open_source_acc_mean": round(float(np.mean(os_accs)), 2),
                    "open_source_acc_std": round(float(np.std(os_accs)), 2),
                    "synthetic_acc_mean": round(float(np.mean(syn_accs)), 2),
                    "synthetic_acc_std": round(float(np.std(syn_accs)), 2),
                },
            },
        }

        # Select overall best checkpoint by validation accuracy
        best_run = max(seed_results, key=lambda x: x["val_acc"])
        best_seed = best_run["seed"]
        src_pt = os.path.join(CKPT_DIR, f"{m_name}_20class_seed{best_seed}_best.pt")
        dst_pt = os.path.join(CKPT_DIR, f"{m_name}_20class_final.pt")
        import shutil
        shutil.copyfile(src_pt, dst_pt)
        summary["selected_final_checkpoint"] = {
            "seed": best_seed,
            "val_acc": best_run["val_acc"],
            "path": dst_pt,
        }

        # Copy validation tau sweep table of selected run to canonical export
        src_sweep = os.path.join(EXPORTS_DIR, f"tau_sweep_val_{m_name}_seed{best_seed}.csv")
        dst_sweep = os.path.join(EXPORTS_DIR, f"tau_sweep_validation_{m_name}.csv")
        if os.path.exists(src_sweep):
            shutil.copyfile(src_sweep, dst_sweep)

        out_summary_file = os.path.join(EXPORTS_DIR, f"eval_summary_{m_name}_20class.json")
        with open(out_summary_file, "w") as f:
            json.dump(summary, f, indent=2)

        all_summaries[m_name] = summary

        print(f"\n✅ {m_name.upper()} 20-CLASS RUN COMPLETED:")
        print(f"   Test Accuracy:        {summary['aggregate_test_cluster']['accuracy_mean']:.2f}% ± {summary['aggregate_test_cluster']['accuracy_std']:.2f}%")
        print(f"   Test Macro F1:        {summary['aggregate_test_cluster']['macro_f1_mean']:.4f} ± {summary['aggregate_test_cluster']['macro_f1_std']:.4f}")
        print(f"   Tuned tau* (mean):    {summary['aggregate_test_cluster']['best_tau_mean']}")
        print(f"   Combined FAR (OOS):   {summary['aggregate_test_cluster']['combined_rule']['far_oos_speech_mean']:.2f}%")
        print(f"   Combined FRR (Filip): {summary['aggregate_test_cluster']['combined_rule']['frr_filipino_group_mean']:.2f}%")
        print(f"   Real Filipino Acc:    {summary['aggregate_test_cluster']['voice_type_breakdown']['real_filipino_acc_mean']:.2f}%")
        print(f"   Final Checkpoint:     Seed {best_seed} -> {dst_pt}")

    with open(os.path.join(EXPORTS_DIR, "v2_20class_side_by_side.json"), "w") as f:
        json.dump(all_summaries, f, indent=2)

    print("\n" + "=" * 70)
    print("🏆 ALL 20-CLASS MULTI-SEED RUNS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()
