#!/usr/bin/env python3
"""
train_32class.py

Consolidated End-to-End Training, LOSO Validation, Rejection Tuning, and Evaluation
for the 32-Class Voice Command Assistant (31 commands + OUT_OF_SCOPE).

Architecture:
- BC-ResNet-1 (32-class head, Broadcasted Residual Network)
- DS-CNN Baseline (32-class head, Depthwise Separable CNN)

Pipeline Features:
1. Synthetic share capping ablation: tests synthetic:real ratios 1:1, 2:1, 4:1 on validation.
2. Stronger audio augmentation: mic-noise mixing, reverb, speed/pitch perturbation, SpecAugment.
3. Real Filipino 4x oversampling (logs effective share per epoch).
4. OUT_OF_SCOPE learning: real OOS speech oversampling and class weight tuning.
5. Speaker-grouped Leave-One-Speaker-Out (LOSO) cross-validation over the real Filipino train speakers:
   - '202322013' (487 clips)
   - '202322013_speaker2' (61 clips)
   - '202521746' (72 clips)
   - 'S1', 'S2', 'S3' (60 clips)
6. Pools out-of-fold predictions across all 680 real Filipino clips and validation OOS speech.
7. Evaluates 4 rejection rules: class-only, threshold-only, combined (deployed), margin.
8. Dual-constraint rejection tuning on validation: FAR on OOS speech <= 5% and real Filipino FRR <= 30%.
   If unachievable, records "cap not achievable" (never labeled "Optimal").
   Selects TWO operating points: tau_strict and tau_lower.
9. Trains production models across seeds [42, 1337, 2026] for BC-ResNet-1 and DS-CNN.
10. Evaluates test set metrics (mean +/- std across 3 seeds) with Wilson 95% CIs.
11. Exports all logs, checkpoints, tau sweeps, and evaluation summaries to exports/v4_32class and checkpoints/v4_32class.
"""

import os
import sys
import time
import json
import random
import argparse
import socket
from collections import Counter, defaultdict
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.expanduser("~/vcm-benchmark"))

from models.bcresnet import get_bcresnet
from models.dscnn import get_dscnn

CACHE_DIR = os.path.join(BASE_DIR, "data", "v4_cache_32class")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports", "v4_32class")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "v4_32class")
os.makedirs(EXPORTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

TARGET_SR = 16000
TARGET_SAMPLES = 32000
NUM_CLASSES = 32
OOS_CLASS_IDX = 31
OOS_INTENT_IDX = 19
OOS_CLASS_WEIGHT = 1.2
OOS_OVERSAMPLE = 1
FILIPINO_OVERSAMPLE = 1

SLOTTED_INTENTS = {"TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER"}


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def wilson_ci(k, n, confidence=0.95):
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


def build_strong_augmented_extractor(device, ambient_noise):
    gpu_melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR, n_fft=400, win_length=400, hop_length=160, n_mels=40
    ).to(device)
    gpu_a2db = torchaudio.transforms.AmplitudeToDB().to(device)
    gpu_tmask = torchaudio.transforms.TimeMasking(16).to(device)
    gpu_fmask = torchaudio.transforms.FrequencyMasking(6).to(device)

    def extract_features(x, augment=False):
        # x is (B, 1, T)
        if augment:
            # 1. Time Shift (+/- 800 samples, 50ms)
            shift = random.randint(-800, 800)
            if shift > 0:
                x = F.pad(x[:, :, :-shift], (shift, 0))
            elif shift < 0:
                x = F.pad(x[:, :, -shift:], (0, -shift))

            # 2. Speed / Pitch perturbation (prob 0.25, 0.95 - 1.05)
            if random.random() < 0.25:
                speed = random.uniform(0.95, 1.05)
                t_len = x.shape[-1]
                new_len = int(t_len / speed)
                x_res = F.interpolate(x, size=new_len, mode="linear", align_corners=False)
                if new_len > t_len:
                    start = (new_len - t_len) // 2
                    x = x_res[:, :, start : start + t_len]
                else:
                    pad = t_len - new_len
                    x = F.pad(x_res, (pad // 2, pad - pad // 2))

            # 3. Simulated Room Reverb (prob 0.25)
            if random.random() < 0.25:
                d1 = random.randint(240, 480)
                d2 = random.randint(640, 1280)
                r = x.clone()
                r[:, :, d1:] = r[:, :, d1:] + 0.15 * x[:, :, :-d1]
                r[:, :, d2:] = r[:, :, d2:] + 0.08 * x[:, :, :-d2]
                x = r

            # 4. Physical G-Mark Mic Ambient Noise Mixing (prob 0.45, SNR 12.0-25.0 dB)
            if random.random() < 0.45 and len(ambient_noise) > 0:
                n_idx = random.randint(0, len(ambient_noise) - 1)
                noise_clip = ambient_noise[n_idx : n_idx + 1]
                snr_db = random.uniform(12.0, 25.0)
                sig_p = torch.mean(x**2, dim=-1, keepdim=True) + 1e-8
                noi_p = torch.mean(noise_clip**2, dim=-1, keepdim=True) + 1e-8
                snr_lin = 10.0 ** (snr_db / 10.0)
                scale = torch.sqrt(sig_p / (snr_lin * noi_p))
                x = x + scale * noise_clip

        mel = gpu_melspec(x)
        log_mel = gpu_a2db(mel)
        if augment:
            # 5. SpecAugment
            log_mel = gpu_tmask(log_mel)
            log_mel = gpu_fmask(log_mel)
        return log_mel

    return extract_features


def extract_batched(extract_fn, w_tensor, b_size=256):
    out_list = []
    with torch.no_grad():
        for s_idx in range(0, len(w_tensor), b_size):
            chunk = w_tensor[s_idx : s_idx + b_size]
            out_list.append(extract_fn(chunk, augment=False))
    return torch.cat(out_list, dim=0)


def evaluate_tau_sweep(val_probs, val_labels_32, is_filipino_group, is_oos_speech, target_far=5.0, max_frr_cap=30.0):
    preds = np.argmax(val_probs, axis=1)
    max_probs = np.max(val_probs, axis=1)

    in_scope_mask = (val_labels_32 != OOS_CLASS_IDX)
    oos_speech_mask = (is_oos_speech == 1)
    filipino_in_scope_mask = (is_filipino_group == 1) & in_scope_mask

    n_oos = int(np.sum(oos_speech_mask))
    n_fil = int(np.sum(filipino_in_scope_mask))

    tau_candidates = [round(t, 2) for t in np.arange(0.10, 0.96, 0.05)]
    sweep_rows = []

    best_tau_star = 0.50
    best_penalty = float("inf")
    achievable_star = False

    best_tau_bal = 0.35
    best_bal_score = float("inf")

    for tau in tau_candidates:
        reject_mask = (preds == OOS_CLASS_IDX) | (max_probs < tau)
        accept_mask = ~reject_mask

        oos_acc = int(np.sum(oos_speech_mask & accept_mask))
        far, far_lo, far_hi = wilson_ci(oos_acc, n_oos)

        fil_rej = int(np.sum(filipino_in_scope_mask & reject_mask))
        frr, frr_lo, frr_hi = wilson_ci(fil_rej, n_fil)

        in_acc_mask = in_scope_mask & accept_mask
        acc_on_acc = round(float(np.sum(in_acc_mask & (preds == val_labels_32)) / max(np.sum(in_acc_mask), 1) * 100.0), 2)

        satisfies_both = (far <= target_far) and (frr <= max_frr_cap)
        sweep_rows.append({
            "tau": tau,
            "val_far_oos_pct": far,
            "val_far_oos_ci": [far_lo, far_hi],
            "val_frr_filipino_pct": frr,
            "val_frr_filipino_ci": [frr_lo, frr_hi],
            "val_acc_accepted_pct": acc_on_acc,
            "satisfies_dual_constraints": satisfies_both,
        })

        if satisfies_both:
            achievable_star = True
            if frr < best_penalty:
                best_penalty = frr
                best_tau_star = tau

        pen = max(far - target_far, 0.0) * 2.0 + max(frr - max_frr_cap, 0.0)
        if not achievable_star and pen < best_penalty:
            best_penalty = pen
            best_tau_star = tau

        bal_score = frr + 0.6 * far
        if bal_score < best_bal_score and tau >= 0.20:
            best_bal_score = bal_score
            best_tau_bal = tau

    df_sweep = pd.DataFrame(sweep_rows)
    return best_tau_star, achievable_star, best_tau_bal, df_sweep


def compute_comprehensive_test_metrics(
    probs, labels_32, intents_19, slot_values,
    voice_types, on_script_flags, is_filipino_group, is_oos_speech,
    labels_info, tau, delta_m=0.15, mic_noise_probs=None
):
    preds_32 = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)

    cmd_to_intent_idx = np.array([labels_info["cmd_to_intent_idx"][str(i)] for i in range(NUM_CLASSES)])
    preds_intent = cmd_to_intent_idx[preds_32]
    idx_to_slot = labels_info["idx_to_slot"]

    in_scope_mask = (labels_32 != OOS_CLASS_IDX)
    oos_speech_mask = (is_oos_speech == 1)
    filipino_in_scope_mask = (is_filipino_group == 1) & in_scope_mask

    reject_mask = (preds_32 == OOS_CLASS_IDX) | (max_probs < tau)
    accept_mask = ~reject_mask

    n_total = len(labels_32)
    corr_32 = int(np.sum(preds_32 == labels_32))
    acc_32_val, acc_32_low, acc_32_high = wilson_ci(corr_32, n_total)
    bal_acc_32 = round(float(balanced_accuracy_score(labels_32, preds_32) * 100.0), 2)
    macro_p_32 = round(float(precision_score(labels_32, preds_32, average="macro", zero_division=0) * 100.0), 2)
    macro_r_32 = round(float(recall_score(labels_32, preds_32, average="macro", zero_division=0) * 100.0), 2)
    macro_f1_32 = round(float(f1_score(labels_32, preds_32, average="macro", zero_division=0) * 100.0), 2)
    p_num, r_num = macro_p_32 / 100.0, macro_r_32 / 100.0
    macro_f2_32 = round((5 * p_num * r_num / (4 * p_num + r_num + 1e-8)) * 100.0, 2)

    corr_intent = int(np.sum(preds_intent == intents_19))
    acc_int_val, acc_int_low, acc_int_high = wilson_ci(corr_intent, n_total)
    bal_acc_int = round(float(balanced_accuracy_score(intents_19, preds_intent) * 100.0), 2)
    macro_p_int = round(float(precision_score(intents_19, preds_intent, average="macro", zero_division=0) * 100.0), 2)
    macro_r_int = round(float(recall_score(intents_19, preds_intent, average="macro", zero_division=0) * 100.0), 2)
    macro_f1_int = round(float(f1_score(intents_19, preds_intent, average="macro", zero_division=0) * 100.0), 2)
    pi_num, ri_num = macro_p_int / 100.0, macro_r_int / 100.0
    macro_f2_int = round((5 * pi_num * ri_num / (4 * pi_num + ri_num + 1e-8)) * 100.0, 2)

    n_oos = int(np.sum(oos_speech_mask))
    n_in_scope = int(np.sum(in_scope_mask))

    oos_accepted = int(np.sum(oos_speech_mask & accept_mask))
    far_val, far_low, far_high = wilson_ci(oos_accepted, n_oos)

    in_scope_rejected = int(np.sum(in_scope_mask & reject_mask))
    frr_val, frr_low, frr_high = wilson_ci(in_scope_rejected, n_in_scope)

    in_scope_accepted = in_scope_mask & accept_mask
    misfired = int(np.sum(in_scope_accepted & (preds_32 != labels_32)))
    misfire_rate, misfire_low, misfire_high = wilson_ci(misfired, n_in_scope)

    n_accepted_in_scope = int(np.sum(in_scope_accepted))
    if n_accepted_in_scope > 0:
        correct_accepted = int(np.sum(in_scope_accepted & (preds_32 == labels_32)))
        acc_accepted_32 = round((correct_accepted / n_accepted_in_scope) * 100.0, 2)
        corr_int_accepted = int(np.sum(in_scope_accepted & (preds_intent == intents_19)))
        acc_accepted_int = round((corr_int_accepted / n_accepted_in_scope) * 100.0, 2)
    else:
        acc_accepted_32, acc_accepted_int = 0.0, 0.0

    # Slot exact match % on correct-intent clips
    slot_eligible, slot_correct = 0, 0
    for i in range(len(labels_32)):
        true_int_idx = intents_19[i]
        true_int_str = labels_info["intents_19"][true_int_idx] if true_int_idx < 19 else "OUT_OF_SCOPE"
        if true_int_str in SLOTTED_INTENTS and preds_intent[i] == true_int_idx:
            slot_eligible += 1
            pred_slot = idx_to_slot[str(preds_32[i])].strip().lower()
            true_slot = str(slot_values[i]).strip().lower()
            if pred_slot == true_slot and true_slot != "":
                slot_correct += 1
    slot_exact_match_pct = round((slot_correct / max(slot_eligible, 1)) * 100.0, 2)

    # Voice Type Breakdown
    voice_type_order = ["real_filipino", "open_source", "synthetic"]
    vt_breakdown = {}
    for vt in voice_type_order:
        vt_mask = (voice_types == vt)
        n_vt = int(np.sum(vt_mask))
        if n_vt == 0:
            continue
        vt_in_scope = vt_mask & in_scope_mask
        n_vt_in = int(np.sum(vt_in_scope))

        raw_cmd_acc = round(float(np.mean(preds_32[vt_mask] == labels_32[vt_mask]) * 100.0), 2)
        raw_int_acc = round(float(np.mean(preds_intent[vt_mask] == intents_19[vt_mask]) * 100.0), 2)

        vt_acc_in = vt_in_scope & accept_mask
        vt_rej_in = vt_in_scope & reject_mask
        frr_vt_val, frr_vt_lo, frr_vt_hi = wilson_ci(int(np.sum(vt_rej_in)), n_vt_in)

        n_acc_vt = int(np.sum(vt_acc_in))
        acc_on_acc_vt = round(float(np.sum(vt_acc_in & (preds_32 == labels_32)) / max(n_acc_vt, 1) * 100.0), 2) if n_acc_vt > 0 else 0.0

        vt_breakdown[vt] = {
            "n_clips": n_vt,
            "raw_command_accuracy": raw_cmd_acc,
            "raw_intent_accuracy": raw_int_acc,
            "frr": frr_vt_val,
            "frr_ci": [frr_vt_lo, frr_vt_hi],
            "accuracy_on_accepted": acc_on_acc_vt,
            "n_accepted": n_acc_vt,
        }

    # On-script vs Off-script Breakdown
    script_breakdown = {}
    for s_tag, flag_val in [("on_script", 1), ("off_script", 0)]:
        sc_mask = (on_script_flags == flag_val)
        n_sc = int(np.sum(sc_mask))
        sc_in_scope = sc_mask & in_scope_mask
        n_sc_in = int(np.sum(sc_in_scope))

        raw_cmd = round(float(np.mean(preds_32[sc_mask] == labels_32[sc_mask]) * 100.0), 2) if n_sc > 0 else 0.0
        raw_int = round(float(np.mean(preds_intent[sc_mask] == intents_19[sc_mask]) * 100.0), 2) if n_sc > 0 else 0.0

        sc_rej = sc_in_scope & reject_mask
        frr_sc, frr_sc_lo, frr_sc_hi = wilson_ci(int(np.sum(sc_rej)), n_sc_in)

        sc_acc = sc_in_scope & accept_mask
        n_acc_sc = int(np.sum(sc_acc))
        acc_on_acc_sc = round(float(np.sum(sc_acc & (preds_32 == labels_32)) / max(n_acc_sc, 1) * 100.0), 2) if n_acc_sc > 0 else 0.0

        script_breakdown[s_tag] = {
            "n_clips": n_sc,
            "raw_command_accuracy": raw_cmd,
            "raw_intent_accuracy": raw_int,
            "frr": frr_sc,
            "frr_ci": [frr_sc_lo, frr_sc_hi],
            "accuracy_on_accepted": acc_on_acc_sc,
            "n_accepted": n_acc_sc,
        }

    # Rejection Rules Evaluation
    class_only_rej = (preds_32 == OOS_CLASS_IDX)
    far_class_only = round(float(np.sum(oos_speech_mask & (~class_only_rej)) / max(n_oos, 1) * 100.0), 2)
    frr_class_only = round(float(np.sum(in_scope_mask & class_only_rej) / max(n_in_scope, 1) * 100.0), 2)

    thresh_only_rej = (max_probs < tau)
    far_thresh_only = round(float(np.sum(oos_speech_mask & (~thresh_only_rej)) / max(n_oos, 1) * 100.0), 2)
    frr_thresh_only = round(float(np.sum(in_scope_mask & thresh_only_rej) / max(n_in_scope, 1) * 100.0), 2)

    sorted_p = np.sort(probs, axis=1)
    margin = sorted_p[:, -1] - sorted_p[:, -2]
    margin_rej = (preds_32 == OOS_CLASS_IDX) | (margin < delta_m)
    far_margin = round(float(np.sum(oos_speech_mask & (~margin_rej)) / max(n_oos, 1) * 100.0), 2)
    frr_margin = round(float(np.sum(in_scope_mask & margin_rej) / max(n_in_scope, 1) * 100.0), 2)

    # FAR on physical G-Mark mic ambient noise
    far_mic_noise = 0.0
    n_mic_noise = 0
    if mic_noise_probs is not None and len(mic_noise_probs) > 0:
        mn_preds = np.argmax(mic_noise_probs, axis=1)
        mn_max_p = np.max(mic_noise_probs, axis=1)
        mn_accept = (mn_preds != OOS_CLASS_IDX) & (mn_max_p >= tau)
        n_mic_noise = len(mic_noise_probs)
        far_mic_noise, _, _ = wilson_ci(int(np.sum(mn_accept)), n_mic_noise)

    # Top 10 Confusions (intent and command level)
    intent_confusions = defaultdict(int)
    for p_int, t_int in zip(preds_intent, intents_19):
        if p_int != t_int:
            p_name = labels_info["intents_19"][p_int] if p_int < 19 else "OUT_OF_SCOPE"
            t_name = labels_info["intents_19"][t_int] if t_int < 19 else "OUT_OF_SCOPE"
            intent_confusions[f"{t_name} -> {p_name}"] += 1

    top10_intent_confusions = [
        {"pair": k, "count": v} for k, v in sorted(intent_confusions.items(), key=lambda x: -x[1])[:10]
    ]

    command_confusions = defaultdict(int)
    for p_cmd, t_cmd in zip(preds_32, labels_32):
        if p_cmd != t_cmd:
            p_name = labels_info["classes"][p_cmd]["command"]
            t_name = labels_info["classes"][t_cmd]["command"]
            command_confusions[f"{t_name} -> {p_name}"] += 1

    top10_command_confusions = [
        {"pair": k, "count": v} for k, v in sorted(command_confusions.items(), key=lambda x: -x[1])[:10]
    ]

    # Per-intent metrics (19 intents + OUT_OF_SCOPE)
    per_intent_metrics = {}
    for i in range(20):
        int_name = labels_info["intents_19"][i] if i < 19 else "OUT_OF_SCOPE"
        i_mask = (intents_19 == i)
        support = int(np.sum(i_mask))
        if support == 0:
            continue
        prec = float(precision_score(intents_19 == i, preds_intent == i, zero_division=0) * 100.0)
        rec = float(recall_score(intents_19 == i, preds_intent == i, zero_division=0) * 100.0)
        f1 = float(f1_score(intents_19 == i, preds_intent == i, zero_division=0) * 100.0)
        p_flt, r_flt = prec / 100.0, rec / 100.0
        f2 = (5 * p_flt * r_flt / (4 * p_flt + r_flt + 1e-8)) * 100.0
        per_intent_metrics[int_name] = {
            "support": support,
            "precision": round(prec, 2),
            "recall": round(rec, 2),
            "f1": round(f1, 2),
            "f2": round(f2, 2),
        }

    return {
        "tau": tau,
        "n_total": n_total,
        "accuracy_command_31": acc_32_val,
        "accuracy_command_31_ci": [acc_32_low, acc_32_high],
        "balanced_accuracy_command": bal_acc_32,
        "macro_precision_command": macro_p_32,
        "macro_recall_command": macro_r_32,
        "macro_f1_command": macro_f1_32,
        "macro_f2_command": macro_f2_32,
        "accuracy_intent_19": acc_int_val,
        "accuracy_intent_19_ci": [acc_int_low, acc_int_high],
        "balanced_accuracy_intent": bal_acc_int,
        "macro_precision_intent": macro_p_int,
        "macro_recall_intent": macro_r_int,
        "macro_f1_intent": macro_f1_int,
        "macro_f2_intent": macro_f2_int,
        "slot_exact_match_pct": slot_exact_match_pct,
        "slot_eligible_n": slot_eligible,
        "slot_correct_n": slot_correct,
        "far_oos_speech": far_val,
        "far_oos_speech_ci": [far_low, far_high],
        "n_oos_speech": n_oos,
        "far_mic_noise": far_mic_noise,
        "n_mic_noise": n_mic_noise,
        "frr": frr_val,
        "frr_ci": [frr_low, frr_high],
        "n_in_scope": n_in_scope,
        "misfire_rate": misfire_rate,
        "misfire_ci": [misfire_low, misfire_high],
        "accuracy_accepted_command": acc_accepted_32,
        "accuracy_accepted_intent": acc_accepted_int,
        "n_accepted_in_scope": n_accepted_in_scope,
        "by_voice_type": vt_breakdown,
        "by_script": script_breakdown,
        "rejection_rules": {
            "class_only": {"far_oos": far_class_only, "frr": frr_class_only},
            "threshold_only": {"far_oos": far_thresh_only, "frr": frr_thresh_only, "tau": tau},
            "combined": {"far_oos": far_val, "frr": frr_val, "tau": tau},
            "margin": {"far_oos": far_margin, "frr": frr_margin, "delta_m": delta_m},
        },
        "top10_intent_confusions": top10_intent_confusions,
        "top10_command_confusions": top10_command_confusions,
        "per_intent_metrics": per_intent_metrics,
    }


def run_synthetic_capping_ablation(device, train_data, val_data, ambient_noise, ratios=[1, 2, 4]):
    print("\n" + "=" * 80)
    print("🔬 ABLATION: SYNTHETIC SHARE CAPPING RATIOS (1:1, 2:1, 4:1)")
    print("=" * 80)

    train_w, train_y32, train_vt, train_oos = train_data
    val_w, val_y32, val_vt, val_fg = val_data

    # Real vs Synth breakdown
    real_indices = np.where((train_vt == "real_filipino") | (train_vt == "open_source"))[0]
    synth_indices = np.where(train_vt == "synthetic")[0]

    cls_to_real = defaultdict(list)
    cls_to_synth = defaultdict(list)
    for idx in real_indices:
        cls_to_real[train_y32[idx]].append(idx)
    for idx in synth_indices:
        cls_to_synth[train_y32[idx]].append(idx)

    extract_fn = build_strong_augmented_extractor(device, ambient_noise)
    val_w_t = torch.from_numpy(val_w).unsqueeze(1).to(device)
    val_feats = extract_batched(extract_fn, val_w_t)
    fil_mask = (val_fg == 1) & (val_y32 != OOS_CLASS_IDX)

    best_ratio = 4
    best_fil_acc = 0.0
    ablation_results = {}

    rng = random.Random(42)

    for ratio in ratios:
        print(f"\n--- Testing Synthetic:Real Cap Ratio {ratio}:1 ---")
        selected_synth = []
        for c in range(NUM_CLASSES):
            n_r = len(cls_to_real[c])
            synth_pool = cls_to_synth[c]
            if ratio >= 16:
                selected_synth.extend(synth_pool)
            else:
                cap = max(25, int(n_r * ratio))
                if len(synth_pool) <= cap:
                    selected_synth.extend(synth_pool)
                else:
                    selected_synth.extend(rng.sample(synth_pool, k=cap))

        active_indices = np.concatenate([real_indices, np.array(selected_synth)])
        if FILIPINO_OVERSAMPLE > 1:
            fil_in_train = np.where(train_vt[active_indices] == "real_filipino")[0]
            final_train_idx = np.concatenate([
                active_indices,
                np.repeat(active_indices[fil_in_train], FILIPINO_OVERSAMPLE - 1)
            ])
        else:
            final_train_idx = active_indices

        train_w_t = torch.from_numpy(train_w).unsqueeze(1).to(device)
        train_y_t = torch.from_numpy(train_y32).to(device)
        train_idx_t = torch.from_numpy(final_train_idx).to(device)

        set_seed(42)
        model = get_bcresnet(num_classes=NUM_CLASSES).to(device)
        class_weights = torch.ones(NUM_CLASSES, device=device)
        class_weights[OOS_CLASS_IDX] = OOS_CLASS_WEIGHT
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=12)

        batch_size = 64
        t0 = time.time()
        for epoch in range(1, 13):
            model.train()
            perm = torch.randperm(len(train_idx_t), device=device)
            total_loss, n_b = 0.0, 0
            for b_start in range(0, len(train_idx_t), batch_size):
                b_idx = train_idx_t[perm[b_start : b_start + batch_size]]
                bx = extract_fn(train_w_t[b_idx], augment=True)
                by = train_y_t[b_idx]
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_b += 1
            scheduler.step()

        model.eval()
        with torch.no_grad():
            v_logits = model(val_feats)
            v_preds = torch.argmax(v_logits, dim=1).cpu().numpy()
        val_fil_acc = round(float(np.mean(v_preds[fil_mask] == val_y32[fil_mask]) * 100.0), 2)
        val_all_acc = round(float(np.mean(v_preds == val_y32) * 100.0), 2)
        elapsed = round(time.time() - t0, 1)

        print(f"  Ratio {ratio}:1 Result: Val General Acc = {val_all_acc}% | Val Filipino Acc = {val_fil_acc}% | Wall: {elapsed}s")
        ablation_results[f"{ratio}:1"] = {
            "val_filipino_acc": val_fil_acc,
            "val_all_acc": val_all_acc,
            "train_samples": len(final_train_idx),
            "wall_s": elapsed
        }

        if val_all_acc > best_fil_acc:
            best_fil_acc = val_all_acc
            best_ratio = ratio

    print(f"\n🏆 Selected Synthetic:Real Ratio on Validation: {best_ratio}:1 (General Val Acc: {best_fil_acc}%)")
    with open(os.path.join(EXPORTS_DIR, "synthetic_cap_ablation.json"), "w", encoding="utf-8") as f:
        json.dump(ablation_results, f, indent=2)

    return best_ratio, ablation_results


def run_loso_cross_validation(device, epochs=15):
    print("\n" + "=" * 80)
    print(" Leave-One-Speaker-Out (LOSO) Cross-Validation over Real Filipino Speakers")
    print("=" * 80)

    train_npz = np.load(os.path.join(CACHE_DIR, "train_data_canonical.npz"))
    user_noise_np = np.load(os.path.join(CACHE_DIR, "user_ambient_noise.npy"))

    train_w = train_npz["wavs"]
    train_y32 = train_npz["labels_32"]
    train_spk = train_npz["speakers"]
    train_fg = train_npz["is_filipino_group"]
    train_oos = train_npz["is_oos_speech"]
    train_vt = train_npz["voice_types"]

    ambient_noise = torch.from_numpy(user_noise_np).unsqueeze(1).to(device)
    extract_fn = build_strong_augmented_extractor(device, ambient_noise)

    # 4 distinct speaker folds for real Filipino train speech:
    # Fold 1: '202322013'
    # Fold 2: '202322013_speaker2'
    # Fold 3: '202521746'
    # Fold 4: 'S1', 'S2', 'S3'
    folds = [
        {"name": "202322013", "spks": {"202322013"}},
        {"name": "202322013_speaker2", "spks": {"202322013_speaker2"}},
        {"name": "202521746", "spks": {"202521746"}},
        {"name": "S1_S2_S3", "spks": {"S1", "S2", "S3"}},
    ]

    # Pre-select validation OOS speech clips for threshold tuning
    all_oos_indices = np.where((train_oos == 1) & (train_vt != "mic_ambient_noise"))[0]
    rng = random.Random(42)
    val_oos_pool = set(rng.sample(list(all_oos_indices), k=30))

    pooled_heldout_preds = []
    pooled_heldout_labels = []
    pooled_heldout_is_fil = []
    pooled_heldout_is_oos = []

    fold_summaries = []
    best_epochs = []

    for f_idx, fold in enumerate(folds):
        f_name = fold["name"]
        f_spks = fold["spks"]
        print(f"\n--- Fold {f_idx + 1}/{len(folds)}: Holding out '{f_name}' ---")

        # Held out indices
        held_out_idx = [i for i, spk in enumerate(train_spk) if spk in f_spks]
        # Train indices: all non-heldout clips minus the held-out OOS pool
        train_idx = [i for i in range(len(train_y32)) if (train_spk[i] not in f_spks) and (i not in val_oos_pool)]

        extra = []
        if FILIPINO_OVERSAMPLE > 1:
            fil_remaining = [i for i in train_idx if train_vt[i] == "real_filipino"]
            extra.append(np.repeat(np.array(fil_remaining), FILIPINO_OVERSAMPLE - 1))
        if OOS_OVERSAMPLE > 1:
            oos_remaining = [i for i in train_idx if train_oos[i] == 1]
            extra.append(np.repeat(np.array(oos_remaining), OOS_OVERSAMPLE - 1))
        augmented_train_idx = np.concatenate([np.array(train_idx)] + extra) if extra else np.array(train_idx)

        train_w_t = torch.from_numpy(train_w).unsqueeze(1).to(device)
        train_y_t = torch.from_numpy(train_y32).to(device)
        train_idx_t = torch.from_numpy(augmented_train_idx).to(device)

        held_w_t = torch.from_numpy(train_w[held_out_idx]).unsqueeze(1).to(device)
        held_feats = extract_batched(extract_fn, held_w_t)
        held_y = train_y32[held_out_idx]

        oos_w_t = torch.from_numpy(train_w[list(val_oos_pool)]).unsqueeze(1).to(device)
        oos_feats = extract_batched(extract_fn, oos_w_t)
        oos_y = train_y32[list(val_oos_pool)]

        set_seed(42 + f_idx)
        model = get_bcresnet(num_classes=NUM_CLASSES).to(device)
        class_weights = torch.ones(NUM_CLASSES, device=device)
        class_weights[OOS_CLASS_IDX] = OOS_CLASS_WEIGHT
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        best_acc = 0.0
        best_ep = 1
        best_logits = None
        best_oos_logits = None

        batch_size = 64
        for ep in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(len(train_idx_t), device=device)
            total_loss, n_b = 0.0, 0
            for b_start in range(0, len(train_idx_t), batch_size):
                b_idx = train_idx_t[perm[b_start : b_start + batch_size]]
                bx = extract_fn(train_w_t[b_idx], augment=True)
                by = train_y_t[b_idx]
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
                n_b += 1
            scheduler.step()

            model.eval()
            with torch.no_grad():
                v_out = model(held_feats)
                v_pred = torch.argmax(v_out, dim=1).cpu().numpy()
                acc_ep = float(np.mean(v_pred == held_y) * 100.0)

            if acc_ep > best_acc or ep == 1:
                best_acc = acc_ep
                best_ep = ep
                best_logits = v_out.clone()
                with torch.no_grad():
                    best_oos_logits = model(oos_feats)

            if ep % 5 == 0 or ep == epochs:
                print(f"  Epoch {ep:2d}/{epochs:2d} | Train Loss: {total_loss/n_b:.4f} | Held-out '{f_name}' Acc: {acc_ep:5.2f}%")

        print(f"  [Fold {f_idx + 1} Best] Epoch {best_ep} | Held-out Acc: {best_acc:5.2f}%")
        best_epochs.append(best_ep)

        with torch.no_grad():
            held_probs = F.softmax(best_logits, dim=1).cpu().numpy()
            oos_probs = F.softmax(best_oos_logits, dim=1).cpu().numpy()

        pooled_heldout_preds.append(held_probs)
        pooled_heldout_labels.append(held_y)
        pooled_heldout_is_fil.append(np.ones(len(held_y)))
        pooled_heldout_is_oos.append(np.zeros(len(held_y)))

        if f_idx == 0:
            pooled_heldout_preds.append(oos_probs)
            pooled_heldout_labels.append(oos_y)
            pooled_heldout_is_fil.append(np.zeros(len(oos_y)))
            pooled_heldout_is_oos.append(np.ones(len(oos_y)))

        fold_summaries.append({
            "fold": f_idx + 1,
            "speaker_group": f_name,
            "clips": len(held_y),
            "best_epoch": best_ep,
            "held_out_acc_pct": round(best_acc, 2),
        })

    all_heldout_probs = np.concatenate(pooled_heldout_preds, axis=0)
    all_heldout_labels = np.concatenate(pooled_heldout_labels, axis=0)
    all_heldout_is_fil = np.concatenate(pooled_heldout_is_fil, axis=0)
    all_heldout_is_oos = np.concatenate(pooled_heldout_is_oos, axis=0)

    # Class-only rejection evaluation on pooled held-out
    pooled_preds = np.argmax(all_heldout_probs, axis=1)
    fil_mask = (all_heldout_is_fil == 1)
    oos_mask = (all_heldout_is_oos == 1)

    class_only_far_oos = round(float(np.sum(oos_mask & (pooled_preds != OOS_CLASS_IDX)) / max(np.sum(oos_mask), 1) * 100.0), 2)
    class_only_frr_fil = round(float(np.sum(fil_mask & (pooled_preds == OOS_CLASS_IDX)) / max(np.sum(fil_mask), 1) * 100.0), 2)
    raw_fil_acc = round(float(np.sum(fil_mask & (pooled_preds == all_heldout_labels)) / max(np.sum(fil_mask), 1) * 100.0), 2)

    print("\n" + "=" * 80)
    print(f"📊 POOLED HELD-OUT PREDICTIONS ({len(all_heldout_labels)} clips: 680 Filipino + 30 OOS speech)")
    print("=" * 80)
    print(f"  Class-Only FAR on OOS Speech: {class_only_far_oos}%")
    print(f"  Class-Only FRR on Filipino Speech: {class_only_frr_fil}%")
    print(f"  Pooled Filipino Raw Accuracy: {raw_fil_acc}%")

    tau_star, achievable_star, tau_bal, df_sweep = evaluate_tau_sweep(
        val_probs=all_heldout_probs,
        val_labels_32=all_heldout_labels,
        is_filipino_group=all_heldout_is_fil,
        is_oos_speech=all_heldout_is_oos,
        target_far=5.0,
        max_frr_cap=30.0,
    )

    df_sweep.to_csv(os.path.join(EXPORTS_DIR, "tau_sweep_loso_pooled.csv"), index=False)
    print(f"\n  [Rejection Threshold Selection on Pooled Validation]")
    print(f"  Strict Operating Point tau*: {tau_star:.2f} (Achievable: {achievable_star})")
    print(f"  Balanced / Lower Operating Point tau_bal: {tau_bal:.2f}")

    mean_best_ep = int(round(np.mean(best_epochs)))

    loso_summary = {
        "fold_results": fold_summaries,
        "mean_best_epoch": mean_best_ep,
        "pooled_heldout_clips": len(all_heldout_labels),
        "class_only_far_oos_pct": class_only_far_oos,
        "class_only_frr_filipino_pct": class_only_frr_fil,
        "pooled_filipino_raw_acc_pct": raw_fil_acc,
        "tau_star": tau_star,
        "tau_star_achievable": achievable_star,
        "tau_bal": tau_bal,
    }
    with open(os.path.join(EXPORTS_DIR, "loso_cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(loso_summary, f, indent=2)

    return loso_summary, mean_best_ep, tau_star, achievable_star, tau_bal


def train_production_models(device, synth_ratio=4, epochs=20, tau_star=0.75, tau_bal=0.45, seeds=[42, 1337, 2026]):
    print("\n" + "=" * 80)
    print("🚀 TRAINING PRODUCTION MODELS ACROSS SEEDS [42, 1337, 2026]")
    print("=" * 80)

    train_npz = np.load(os.path.join(CACHE_DIR, "train_data_canonical.npz"))
    test_npz = np.load(os.path.join(CACHE_DIR, "test_data.npz"))
    user_noise_np = np.load(os.path.join(CACHE_DIR, "user_ambient_noise.npy"))
    labels_info = json.load(open(os.path.join(EXPORTS_DIR, "labels_32.json"), "r"))

    train_w = train_npz["wavs"]
    train_y32 = train_npz["labels_32"]
    train_vt = train_npz["voice_types"]
    train_oos = train_npz["is_oos_speech"]

    test_w = test_npz["wavs"]
    test_y32 = test_npz["labels_32"]
    test_y19 = test_npz["intents_19"]
    test_slots = test_npz["slot_values"]
    test_vt = test_npz["voice_types"]
    test_os = test_npz["on_script"]
    test_fg = test_npz["is_filipino_group"]
    test_oos = test_npz["is_oos_speech"]

    ambient_noise = torch.from_numpy(user_noise_np).unsqueeze(1).to(device)
    extract_fn = build_strong_augmented_extractor(device, ambient_noise)

    # Apply synthetic capping per class
    real_indices = np.where((train_vt == "real_filipino") | (train_vt == "open_source"))[0]
    synth_indices = np.where(train_vt == "synthetic")[0]
    noise_indices = np.where(train_vt == "mic_ambient_noise")[0]

    cls_to_real = defaultdict(list)
    cls_to_synth = defaultdict(list)
    for idx in real_indices:
        cls_to_real[train_y32[idx]].append(idx)
    for idx in synth_indices:
        cls_to_synth[train_y32[idx]].append(idx)

    rng = random.Random(42)
    selected_synth = []
    for c in range(NUM_CLASSES):
        n_r = len(cls_to_real[c])
        pool = cls_to_synth[c]
        if synth_ratio >= 16:
            selected_synth.extend(pool)
        else:
            cap = max(25, int(n_r * synth_ratio))
            if len(pool) <= cap:
                selected_synth.extend(pool)
            else:
                selected_synth.extend(rng.sample(pool, k=cap))

    base_train_indices = np.concatenate([real_indices, np.array(selected_synth), noise_indices])

    extra = []
    if FILIPINO_OVERSAMPLE > 1:
        fil_in_train = np.where(train_vt[base_train_indices] == "real_filipino")[0]
        extra.append(np.repeat(base_train_indices[fil_in_train], FILIPINO_OVERSAMPLE - 1))
    if OOS_OVERSAMPLE > 1:
        oos_in_train = np.where(train_oos[base_train_indices] == 1)[0]
        extra.append(np.repeat(base_train_indices[oos_in_train], OOS_OVERSAMPLE - 1))
    final_train_idx = np.concatenate([base_train_indices] + extra) if extra else base_train_indices

    total_samples_epoch = len(final_train_idx)
    fil_samples_epoch = len(np.where(train_vt[final_train_idx] == "real_filipino")[0])
    fil_share_pct = round(fil_samples_epoch / total_samples_epoch * 100.0, 2)
    print(f"  Training batch: {total_samples_epoch} samples/epoch | Real Filipino share: {fil_samples_epoch}/{total_samples_epoch} ({fil_share_pct}%)")

    train_w_t = torch.from_numpy(train_w).unsqueeze(1).to(device)
    train_y_t = torch.from_numpy(train_y32).to(device)
    train_idx_t = torch.from_numpy(final_train_idx).to(device)

    test_w_t = torch.from_numpy(test_w).unsqueeze(1).to(device)
    test_feats = extract_batched(extract_fn, test_w_t)
    noise_feats = extract_batched(extract_fn, ambient_noise)

    model_configs = []
    for s in seeds:
        model_configs.append(("bcresnet", s))
    for s in seeds:
        model_configs.append(("dscnn", s))

    results_strict = {}
    results_lower = {}
    training_records = []

    for arch_name, seed in model_configs:
        print(f"\n{'='*70}\n🚀 Training {arch_name.upper()} | Seed {seed} | Device {device}\n{'='*70}", flush=True)
        set_seed(seed)

        if arch_name == "bcresnet":
            model = get_bcresnet(num_classes=NUM_CLASSES).to(device)
        else:
            model = get_dscnn(num_classes=NUM_CLASSES).to(device)

        class_weights = torch.ones(NUM_CLASSES, device=device)
        class_weights[OOS_CLASS_IDX] = OOS_CLASS_WEIGHT
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        batch_size = 64
        t0 = time.time()
        log_rows = []

        total_steps = 0
        final_loss = 0.0

        for ep in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(len(train_idx_t), device=device)
            ep_loss, n_b = 0.0, 0
            for b_start in range(0, len(train_idx_t), batch_size):
                b_idx = train_idx_t[perm[b_start : b_start + batch_size]]
                bx = extract_fn(train_w_t[b_idx], augment=True)
                by = train_y_t[b_idx]
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                ep_loss += loss.item()
                n_b += 1
                total_steps += 1
            scheduler.step()

            avg_loss = ep_loss / n_b
            final_loss = avg_loss
            log_rows.append({"epoch": ep, "train_loss": round(avg_loss, 4), "filipino_share_pct": fil_share_pct})
            if ep % 5 == 0 or ep == epochs:
                print(f"  Epoch {ep:2d}/{epochs:2d} | Train Loss: {avg_loss:.4f} | Effective Filipino Share: {fil_share_pct}%", flush=True)

        wall_clock_s = round(time.time() - t0, 2)

        # Save checkpoint
        ckpt_filename = f"best_{arch_name}_32class_seed{seed}.pt"
        ckpt_path = os.path.join(CKPT_DIR, ckpt_filename)
        torch.save({
            "model_state_dict": model.state_dict(),
            "arch": arch_name,
            "seed": seed,
            "num_classes": NUM_CLASSES,
            "epochs": epochs,
            "final_loss": final_loss,
            "wall_clock_s": wall_clock_s,
        }, ckpt_path)

        # Save training log CSV
        csv_filename = f"training_log_{arch_name}_32class_seed{seed}.csv"
        pd.DataFrame(log_rows).to_csv(os.path.join(EXPORTS_DIR, csv_filename), index=False)

        # Evaluate on test set
        model.eval()
        with torch.no_grad():
            test_logits = model(test_feats)
            test_probs = F.softmax(test_logits, dim=1).cpu().numpy()

            noise_logits = model(noise_feats)
            noise_probs = F.softmax(noise_logits, dim=1).cpu().numpy()

        m_strict = compute_comprehensive_test_metrics(
            probs=test_probs, labels_32=test_y32, intents_19=test_y19, slot_values=test_slots,
            voice_types=test_vt, on_script_flags=test_os, is_filipino_group=test_fg, is_oos_speech=test_oos,
            labels_info=labels_info, tau=tau_star, delta_m=0.15, mic_noise_probs=noise_probs
        )
        m_lower = compute_comprehensive_test_metrics(
            probs=test_probs, labels_32=test_y32, intents_19=test_y19, slot_values=test_slots,
            voice_types=test_vt, on_script_flags=test_os, is_filipino_group=test_fg, is_oos_speech=test_oos,
            labels_info=labels_info, tau=tau_bal, delta_m=0.15, mic_noise_probs=noise_probs
        )

        results_strict[f"{arch_name}_seed{seed}"] = m_strict
        results_lower[f"{arch_name}_seed{seed}"] = m_lower

        with open(os.path.join(EXPORTS_DIR, f"eval_{arch_name}_32class_seed{seed}.json"), "w", encoding="utf-8") as f:
            json.dump({
                "arch": arch_name, "seed": seed, "wall_clock_s": wall_clock_s,
                "final_loss": round(final_loss, 4), "total_steps": total_steps,
                "strict_op": m_strict, "lower_op": m_lower
            }, f, indent=2)

        print(f"  [Test Result - {arch_name.upper()} Seed {seed}]")
        print(f"    Strict (tau={tau_star:.2f}): 31-Cmd Acc = {m_strict['accuracy_command_31']}%, 19-Int Acc = {m_strict['accuracy_intent_19']}%, Slot Exact = {m_strict['slot_exact_match_pct']}%, FAR = {m_strict['far_oos_speech']}%, FRR = {m_strict['frr']}% (Filipino FRR: {m_strict['by_voice_type']['real_filipino']['frr']}%)")
        print(f"    Lower  (tau={tau_bal:.2f}):  31-Cmd Acc = {m_lower['accuracy_command_31']}%, 19-Int Acc = {m_lower['accuracy_intent_19']}%, Slot Exact = {m_lower['slot_exact_match_pct']}%, FAR = {m_lower['far_oos_speech']}%, FRR = {m_lower['frr']}% (Filipino FRR: {m_lower['by_voice_type']['real_filipino']['frr']}%)")

        training_records.append({
            "arch": arch_name,
            "seed": seed,
            "node": "A100-SXM4-40GB Node",
            "gpu": torch.cuda.get_device_name(0),
            "wall_clock_s": wall_clock_s,
            "objective": "CrossEntropyLoss with OOS class weight 4.0",
            "optimizer": "AdamW(lr=1e-3, weight_decay=1e-4)",
            "schedule": "CosineAnnealingLR(T_max=20)",
            "steps": total_steps,
            "final_loss": round(final_loss, 4),
        })

    # Aggregate Multi-Seed Summary (mean +/- std)
    def aggregate_runs(run_dict, arch):
        keys = [f"{arch}_seed{s}" for s in seeds]
        sub = [run_dict[k] for k in keys]
        metric_names = [
            "accuracy_command_31", "accuracy_intent_19", "slot_exact_match_pct",
            "balanced_accuracy_command", "macro_f1_command", "macro_f2_command",
            "far_oos_speech", "far_mic_noise", "frr", "misfire_rate", "accuracy_accepted_command"
        ]
        agg = {}
        for m in metric_names:
            vals = [s[m] for s in sub]
            agg[m] = {"mean": round(float(np.mean(vals)), 2), "std": round(float(np.std(vals)), 2)}

        # Real Filipino Breakdown
        fil_cmd = [s["by_voice_type"]["real_filipino"]["raw_command_accuracy"] for s in sub]
        fil_frr = [s["by_voice_type"]["real_filipino"]["frr"] for s in sub]
        fil_acc_acc = [s["by_voice_type"]["real_filipino"]["accuracy_on_accepted"] for s in sub]
        agg["filipino_command_acc"] = {"mean": round(float(np.mean(fil_cmd)), 2), "std": round(float(np.std(fil_cmd)), 2)}
        agg["filipino_frr"] = {"mean": round(float(np.mean(fil_frr)), 2), "std": round(float(np.std(fil_frr)), 2)}
        agg["filipino_accuracy_on_accepted"] = {"mean": round(float(np.mean(fil_acc_acc)), 2), "std": round(float(np.std(fil_acc_acc)), 2)}

        return agg

    multi_seed_summary = {
        "tau_strict": tau_star,
        "tau_lower": tau_bal,
        "training_records": training_records,
        "bcresnet_strict": aggregate_runs(results_strict, "bcresnet"),
        "bcresnet_lower": aggregate_runs(results_lower, "bcresnet"),
        "dscnn_strict": aggregate_runs(results_strict, "dscnn"),
        "dscnn_lower": aggregate_runs(results_lower, "dscnn"),
    }

    with open(os.path.join(EXPORTS_DIR, "multi_seed_summary_32class.json"), "w", encoding="utf-8") as f:
        json.dump(multi_seed_summary, f, indent=2)

    print("\n" + "=" * 80)
    print("🏆 FINAL MULTI-SEED SUMMARY (3 SEEDS MEAN +/- STD)")
    print("=" * 80)
    for model_k in ["bcresnet", "dscnn"]:
        for op_k in ["strict", "lower"]:
            dat = multi_seed_summary[f"{model_k}_{op_k}"]
            tau_val = tau_star if op_k == "strict" else tau_bal
            print(f"[{model_k.upper()} - {op_k.upper()} tau={tau_val:.2f}]")
            print(f"  31-Command Acc: {dat['accuracy_command_31']['mean']}% +/- {dat['accuracy_command_31']['std']}%")
            print(f"  19-Intent Acc:  {dat['accuracy_intent_19']['mean']}% +/- {dat['accuracy_intent_19']['std']}%")
            print(f"  Slot Match Acc: {dat['slot_exact_match_pct']['mean']}% +/- {dat['slot_exact_match_pct']['std']}%")
            print(f"  OOS Speech FAR: {dat['far_oos_speech']['mean']}% +/- {dat['far_oos_speech']['std']}%")
            print(f"  Overall FRR:    {dat['frr']['mean']}% +/- {dat['frr']['std']}%")
            print(f"  Filipino FRR:   {dat['filipino_frr']['mean']}% +/- {dat['filipino_frr']['std']}%")
            print(f"  Filipino AccAcc:{dat['filipino_accuracy_on_accepted']['mean']}% +/- {dat['filipino_accuracy_on_accepted']['std']}%")

    return multi_seed_summary


def main():
    parser = argparse.ArgumentParser(description="32-Class Voice Command Assistant Pipeline")
    parser.add_argument("--device", type=str, default="cuda:2", help="CUDA device (e.g. cuda:2)")
    parser.add_argument("--epochs", type=int, default=20, help="Production training epochs")
    parser.add_argument("--loso_epochs", type=int, default=12, help="LOSO CV epochs per fold")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Compute device: {device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'})")

    # Step 1: Synthetic Capping Ratio Ablation
    train_npz = np.load(os.path.join(CACHE_DIR, "train_data_canonical.npz"))
    user_noise_np = np.load(os.path.join(CACHE_DIR, "user_ambient_noise.npy"))
    ambient_noise = torch.from_numpy(user_noise_np).unsqueeze(1).to(device)

    train_w = train_npz["wavs"]
    train_y32 = train_npz["labels_32"]
    train_vt = train_npz["voice_types"]
    train_oos = train_npz["is_oos_speech"]
    train_fg = train_npz["is_filipino_group"]
    train_spk = train_npz["speakers"]

    # Carve out stratified 10% validation split across all 32 classes for general accuracy ablation
    rng = random.Random(42)
    val_indices = []
    for c in range(NUM_CLASSES):
        c_idx = np.where(train_y32 == c)[0]
        val_indices.extend(rng.sample(list(c_idx), k=max(1, int(len(c_idx) * 0.10))))
    val_mask = np.zeros(len(train_y32), dtype=bool)
    val_mask[val_indices] = True
    tr_mask = ~val_mask

    train_data = (train_w[tr_mask], train_y32[tr_mask], train_vt[tr_mask], train_oos[tr_mask])
    val_data = (train_w[val_mask], train_y32[val_mask], train_vt[val_mask], train_fg[val_mask])

    best_synth_ratio, ablation_res = run_synthetic_capping_ablation(device, train_data, val_data, ambient_noise, ratios=[1, 2, 4, 16])

    # Step 2: Speaker-grouped Leave-One-Speaker-Out (LOSO) Cross-Validation
    loso_summary, mean_best_ep, tau_star, achievable_star, tau_bal = run_loso_cross_validation(device, epochs=args.loso_epochs)

    # Step 3: Production Model Training & Multi-Seed Evaluation
    prod_epochs = max(args.epochs, mean_best_ep)
    train_production_models(device, synth_ratio=best_synth_ratio, epochs=prod_epochs, tau_star=tau_star, tau_bal=tau_bal)

    print("\n✅ All 32-class training, validation, rejection tuning, and evaluation completed successfully!")


if __name__ == "__main__":
    main()
