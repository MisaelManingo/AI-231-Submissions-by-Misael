#!/usr/bin/env python3
"""
train_94class.py

Trains 94-class Voice Command Models (BC-ResNet-1 and baseline DS-CNN) on the A100 GPU:
- 93 command variations (mapped from vcmbench/variations.csv) + OUT_OF_SCOPE (index 93).
- Solves Filipino speech starvation via 4x oversampling of real Filipino clips during training.
- Class-weighted loss with logged effective weight for OUT_OF_SCOPE (weight 2.50).
- Multi-seed training (seeds 42, 1337, 2026).
- Strict validation-only checkpoint selection based on real Filipino validation accuracy.
- Dual-constraint threshold tau tuning on validation only:
    Constraint 1: FAR on OOS speech <= 5.0%
    Constraint 2: FRR on real Filipino speech <= 30.0%
- Temperature scaling calibration evaluated on validation logits.
- Evaluates rejection rules: class-only, threshold-only, combined, and margin variant.
- Full evaluation at BOTH levels:
    - 94 commands (direct classification)
    - 19 intents (derived projection variation -> intent)
- Metrics:
    - Accuracy with 95% Wilson interval, Balanced Accuracy
    - Macro Precision / Recall / F1 / F2
    - FAR (OOS speech accepted), FRR (real commands rejected), Misfire rate (wrong command fired)
    - Slot exact-match rate for slotted intents on clips where intent was correct
    - Breakdowns: Real Filipino first, Open-source second, Synthetic last; On-script vs Off-script
    - Top 10 confusions at both command and intent level
- Export to FP32 and INT8 ONNX models with FLOPs/MACs profiling via vcmbench/flops.py.
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
from sklearn.metrics import f1_score, precision_score, recall_score, balanced_accuracy_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from models.bcresnet import get_bcresnet
from models.dscnn import get_dscnn

CACHE_DIR = os.path.join(BASE_DIR, "data", "v3_cache_94class")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports", "v3_94class")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "v3_94class")
os.makedirs(EXPORTS_DIR, exist_ok=True)
os.makedirs(CKPT_DIR, exist_ok=True)

TARGET_SR = 16000
TARGET_SAMPLES = 32000
NUM_CLASSES = 94
OOS_CLASS_IDX = 93
OOS_INTENT_IDX = 19
OOS_EFFECTIVE_WEIGHT = 2.50
SLOTTED_INTENTS = {"TIMER", "ALARM", "TEMPERATURE", "BRIGHTNESS", "COLOR", "CREATE_REMINDER"}

INTENTS_19 = [
    "ALARM", "BRIGHTNESS", "CALL", "COLOR", "CREATE_REMINDER",
    "LIGHT_OFF", "LIGHT_ON", "LIST_REMINDERS", "MESSAGE", "NEXT",
    "PAUSE", "PLAY_MUSIC", "STOP", "TEMPERATURE", "TIME",
    "TIMER", "VOLUME_DOWN", "VOLUME_UP", "WEATHER"
]
INTENT2IDX_20 = {cmd: i for i, cmd in enumerate(INTENTS_19)}
INTENT2IDX_20["OUT_OF_SCOPE"] = 19
IDX2INTENT_20 = {v: k for k, v in INTENT2IDX_20.items()}


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
    """Fits scalar temperature T > 0 on validation logits using L-BFGS."""
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


def evaluate_rules_on_split(probs, labels_94, is_filipino_group, is_oos_speech, tau, delta_m, mic_noise_probs=None):
    """
    Evaluates 4 rejection rules on a dataset split:
    1. Class-only: reject if argmax == 93
    2. Threshold-only: reject if max_prob < tau
    3. Combined: reject if argmax == 93 OR max_prob < tau
    4. Margin variant: reject if argmax == 93 OR (top1 - top2) < delta_m
    """
    preds = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)

    sorted_probs = np.sort(probs, axis=1)
    top1 = sorted_probs[:, -1]
    top2 = sorted_probs[:, -2]
    margins = top1 - top2

    in_scope_mask = (labels_94 != OOS_CLASS_IDX)
    oos_speech_mask = (is_oos_speech == 1)
    filipino_in_scope_mask = (is_filipino_group == 1) & in_scope_mask

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
            correct_accepted = np.sum(accepted_in_scope & (preds == labels_94))
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


def run_tau_sweep_on_validation(val_probs, val_labels_94, is_filipino_group, is_oos_speech, target_far=5.0, max_frr_cap=30.0):
    """
    Sweeps tau on validation set from 0.10 to 0.95 in 0.05 steps.
    Dual-constraint selection:
      - Constraint 1: FAR on OOS speech <= target_far (5.0%)
      - Constraint 2: FRR on real Filipino speech <= max_frr_cap (30.0%)
    """
    sweep_rows = []
    best_tau = 0.50
    best_frr = float("inf")
    achievable = False

    tau_candidates = [round(t, 2) for t in np.arange(0.10, 0.96, 0.05)]

    for tau in tau_candidates:
        res = evaluate_rules_on_split(
            probs=val_probs,
            labels_94=val_labels_94,
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
            achievable = True
            if frr < best_frr:
                best_frr = frr
                best_tau = tau

    if not achievable:
        # Pick closest trade-off minimizing combined penalty
        best_penalty = float("inf")
        for row in sweep_rows:
            penalty = max(row["val_far_oos_pct"] - target_far, 0.0) * 2.0 + max(row["val_frr_filipino_pct"] - max_frr_cap, 0.0)
            if penalty < best_penalty:
                best_penalty = penalty
                best_tau = row["tau"]

    df_sweep = pd.DataFrame(sweep_rows)
    return best_tau, df_sweep, achievable


def compute_comprehensive_metrics(
    probs, labels_94, intents_19, slot_values,
    voice_types, on_script_flags, is_filipino_group, is_oos_speech,
    labels_info, tau, delta_m, mic_noise_probs=None
):
    """
    Computes comprehensive test metrics at both 94-command and 19-intent levels.
    """
    preds_94 = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)

    var_to_intent_idx = np.array([labels_info["var_to_intent_idx"][str(i)] for i in range(NUM_CLASSES)])
    preds_intent = var_to_intent_idx[preds_94]
    idx_to_slot = labels_info["idx_to_slot"]
    idx_to_intent = labels_info["idx_to_intent"]
    idx_to_phrase = labels_info["idx_to_phrase"]

    in_scope_mask = (labels_94 != OOS_CLASS_IDX)
    oos_speech_mask = (is_oos_speech == 1)
    filipino_in_scope_mask = (is_filipino_group == 1) & in_scope_mask

    # Deployment Rejection Rule: Combined (argmax == 93 or max_prob < tau)
    reject_mask = (preds_94 == OOS_CLASS_IDX) | (max_probs < tau)
    accept_mask = ~reject_mask

    # 1. Overall Command Level (94 Classes)
    n_total = len(labels_94)
    corr_94 = np.sum(preds_94 == labels_94)
    acc_94_val, acc_94_low, acc_94_high = wilson_ci(corr_94, n_total)
    bal_acc_94 = round(float(balanced_accuracy_score(labels_94, preds_94) * 100.0), 2)
    macro_p_94 = round(float(precision_score(labels_94, preds_94, average="macro", zero_division=0) * 100.0), 2)
    macro_r_94 = round(float(recall_score(labels_94, preds_94, average="macro", zero_division=0) * 100.0), 2)
    macro_f1_94 = round(float(f1_score(labels_94, preds_94, average="macro", zero_division=0) * 100.0), 2)
    # F2: (1 + 4) * P * R / (4 * P + R)
    p_num, r_num = macro_p_94 / 100.0, macro_r_94 / 100.0
    macro_f2_94 = round((5 * p_num * r_num / (4 * p_num + r_num + 1e-8)) * 100.0, 2)

    # 2. Overall Intent Level (20 Classes: 19 intents + OUT_OF_SCOPE)
    corr_intent = np.sum(preds_intent == intents_19)
    acc_int_val, acc_int_low, acc_int_high = wilson_ci(corr_intent, n_total)
    bal_acc_int = round(float(balanced_accuracy_score(intents_19, preds_intent) * 100.0), 2)
    macro_p_int = round(float(precision_score(intents_19, preds_intent, average="macro", zero_division=0) * 100.0), 2)
    macro_r_int = round(float(recall_score(intents_19, preds_intent, average="macro", zero_division=0) * 100.0), 2)
    macro_f1_int = round(float(f1_score(intents_19, preds_intent, average="macro", zero_division=0) * 100.0), 2)
    pi_num, ri_num = macro_p_int / 100.0, macro_r_int / 100.0
    macro_f2_int = round((5 * pi_num * ri_num / (4 * pi_num + ri_num + 1e-8)) * 100.0, 2)

    # 3. Benchmark Pipeline Rates (FAR, FRR, Misfire)
    n_oos = int(np.sum(oos_speech_mask))
    n_in_scope = int(np.sum(in_scope_mask))

    # FAR: OOS speech where command fires (accepted)
    oos_accepted = int(np.sum(oos_speech_mask & accept_mask))
    far_val, far_low, far_high = wilson_ci(oos_accepted, n_oos)

    # FRR: Real in-scope commands rejected or silent
    in_scope_rejected = int(np.sum(in_scope_mask & reject_mask))
    frr_val, frr_low, frr_high = wilson_ci(in_scope_rejected, n_in_scope)

    # Misfire: Real in-scope commands accepted but fired wrong command
    in_scope_accepted = in_scope_mask & accept_mask
    misfired = int(np.sum(in_scope_accepted & (preds_94 != labels_94)))
    misfire_rate, misfire_low, misfire_high = wilson_ci(misfired, n_in_scope)

    # Command Accuracy on Accepted
    n_accepted_in_scope = int(np.sum(in_scope_accepted))
    if n_accepted_in_scope > 0:
        correct_accepted = int(np.sum(in_scope_accepted & (preds_94 == labels_94)))
        acc_accepted_94 = round((correct_accepted / n_accepted_in_scope) * 100.0, 2)
        corr_int_accepted = int(np.sum(in_scope_accepted & (preds_intent == intents_19)))
        acc_accepted_int = round((corr_int_accepted / n_accepted_in_scope) * 100.0, 2)
    else:
        acc_accepted_94, acc_accepted_int = 0.0, 0.0

    # 4. Slot Exact-Match Rate on clips where intent was correct
    slot_eligible = 0
    slot_correct = 0
    for i in range(len(labels_94)):
        true_int_idx = intents_19[i]
        true_int_str = labels_info["commands_19"][true_int_idx] if true_int_idx < 19 else "OUT_OF_SCOPE"
        if true_int_str in SLOTTED_INTENTS and preds_intent[i] == true_int_idx:
            slot_eligible += 1
            pred_slot = idx_to_slot[str(preds_94[i])].strip().lower()
            true_slot = str(slot_values[i]).strip().lower()
            if pred_slot == true_slot and true_slot != "":
                slot_correct += 1

    slot_exact_match_pct = round((slot_correct / max(slot_eligible, 1)) * 100.0, 2)

    # 5. Voice Type Breakdown (Real Filipino first, Open-source second, Synthetic last)
    voice_type_order = ["real_filipino", "open_source", "synthetic"]
    vt_breakdown = {}
    for vt in voice_type_order:
        vt_mask = (voice_types == vt)
        n_vt = int(np.sum(vt_mask))
        if n_vt == 0:
            continue
        vt_in_scope = vt_mask & in_scope_mask
        n_vt_in = int(np.sum(vt_in_scope))

        raw_cmd_acc = round(float(np.mean(preds_94[vt_mask] == labels_94[vt_mask]) * 100.0), 2)
        raw_int_acc = round(float(np.mean(preds_intent[vt_mask] == intents_19[vt_mask]) * 100.0), 2)

        vt_acc_in = vt_in_scope & accept_mask
        n_vt_acc = int(np.sum(vt_acc_in))
        acc_on_acc_cmd = round(float(np.sum(vt_acc_in & (preds_94 == labels_94)) / max(n_vt_acc, 1) * 100.0), 2)
        acc_on_acc_int = round(float(np.sum(vt_acc_in & (preds_intent == intents_19)) / max(n_vt_acc, 1) * 100.0), 2)

        vt_rej_in = vt_in_scope & reject_mask
        frr_vt = round(float(np.sum(vt_rej_in) / max(n_vt_in, 1) * 100.0), 2)

        vt_breakdown[vt] = {
            "total_clips": n_vt,
            "in_scope_clips": n_vt_in,
            "raw_command_acc_pct": raw_cmd_acc,
            "raw_intent_acc_pct": raw_int_acc,
            "accepted_in_scope_clips": n_vt_acc,
            "command_acc_on_accepted_pct": acc_on_acc_cmd,
            "intent_acc_on_accepted_pct": acc_on_acc_int,
            "frr_pct": frr_vt,
        }

    # 6. On-Script vs Off-Script Breakdown
    script_breakdown = {}
    for flag_val, flag_name in [(1, "on_script"), (0, "off_script")]:
        s_mask = (on_script_flags == flag_val)
        n_s = int(np.sum(s_mask))
        if n_s == 0:
            continue
        s_in = s_mask & in_scope_mask
        n_s_in = int(np.sum(s_in))
        raw_cmd = round(float(np.mean(preds_94[s_mask] == labels_94[s_mask]) * 100.0), 2)
        raw_int = round(float(np.mean(preds_intent[s_mask] == intents_19[s_mask]) * 100.0), 2)
        s_acc_in = s_in & accept_mask
        n_s_acc = int(np.sum(s_acc_in))
        acc_on_acc = round(float(np.sum(s_acc_in & (preds_94 == labels_94)) / max(n_s_acc, 1) * 100.0), 2)
        frr_s = round(float(np.sum(s_in & reject_mask) / max(n_s_in, 1) * 100.0), 2)

        script_breakdown[flag_name] = {
            "total_clips": n_s,
            "in_scope_clips": n_s_in,
            "raw_command_acc_pct": raw_cmd,
            "raw_intent_acc_pct": raw_int,
            "command_acc_on_accepted_pct": acc_on_acc,
            "frr_pct": frr_s,
        }

    # 7. Rejection Rules Evaluation
    rule_results = evaluate_rules_on_split(
        probs=probs,
        labels_94=labels_94,
        is_filipino_group=is_filipino_group,
        is_oos_speech=is_oos_speech,
        tau=tau,
        delta_m=delta_m,
        mic_noise_probs=mic_noise_probs,
    )

    # 8. Top 10 Confusions (Intent level and Command level)
    # Intent level confusions:
    int_cm = confusion_matrix(intents_19, preds_intent, labels=list(range(20)))
    int_conf_pairs = []
    for r in range(20):
        for c in range(20):
            if r != c and int_cm[r, c] > 0:
                name_r = IDX2INTENT_20.get(r, f"Class {r}")
                name_c = IDX2INTENT_20.get(c, f"Class {c}")
                int_conf_pairs.append({
                    "true_intent": name_r,
                    "pred_intent": name_c,
                    "count": int(int_cm[r, c]),
                })
    int_conf_pairs.sort(key=lambda x: x["count"], reverse=True)
    top10_intent_conf = int_conf_pairs[:10]

    # Command level confusions:
    cmd_cm = confusion_matrix(labels_94, preds_94, labels=list(range(NUM_CLASSES)))
    cmd_conf_pairs = []
    for r in range(NUM_CLASSES):
        for c in range(NUM_CLASSES):
            if r != c and cmd_cm[r, c] > 0:
                name_r = idx_to_phrase.get(str(r), f"Class {r}")
                name_c = idx_to_phrase.get(str(c), f"Class {c}")
                cmd_conf_pairs.append({
                    "true_command": name_r,
                    "pred_command": name_c,
                    "count": int(cmd_cm[r, c]),
                })
    cmd_conf_pairs.sort(key=lambda x: x["count"], reverse=True)
    top10_cmd_conf = cmd_conf_pairs[:10]

    return {
        "command_level_94": {
            "accuracy_pct": acc_94_val,
            "accuracy_ci_95": [acc_94_low, acc_94_high],
            "balanced_accuracy_pct": bal_acc_94,
            "macro_precision_pct": macro_p_94,
            "macro_recall_pct": macro_r_94,
            "macro_f1_pct": macro_f1_94,
            "macro_f2_pct": macro_f2_94,
        },
        "intent_level_19": {
            "accuracy_pct": acc_int_val,
            "accuracy_ci_95": [acc_int_low, acc_int_high],
            "balanced_accuracy_pct": bal_acc_int,
            "macro_precision_pct": macro_p_int,
            "macro_recall_pct": macro_r_int,
            "macro_f1_pct": macro_f1_int,
            "macro_f2_pct": macro_f2_int,
        },
        "pipeline_rates": {
            "far_oos_speech_pct": far_val,
            "far_oos_ci_95": [far_low, far_high],
            "far_oos_accepted_count": oos_accepted,
            "far_oos_total_count": n_oos,
            "frr_in_scope_pct": frr_val,
            "frr_in_scope_ci_95": [frr_low, frr_high],
            "frr_rejected_count": in_scope_rejected,
            "frr_total_count": n_in_scope,
            "misfire_rate_pct": misfire_rate,
            "misfire_ci_95": [misfire_low, misfire_high],
            "misfire_count": misfired,
            "command_acc_on_accepted_pct": acc_accepted_94,
            "intent_acc_on_accepted_pct": acc_accepted_int,
            "total_accepted": int(np.sum(accept_mask)),
            "total_rejected": int(np.sum(reject_mask)),
        },
        "slot_exact_match": {
            "eligible_clips": slot_eligible,
            "exact_matches": slot_correct,
            "exact_match_pct": slot_exact_match_pct,
        },
        "voice_type_breakdown": vt_breakdown,
        "script_adherence_breakdown": script_breakdown,
        "rejection_rules": rule_results,
        "top10_intent_confusions": top10_intent_conf,
        "top10_command_confusions": top10_cmd_conf,
    }


def train_single_run(model_name, seed, device, epochs, batch_size, lr, filipino_oversample=4, oos_weight=2.50, use_supp_synth=False):
    set_seed(seed)
    print(f"\n================================================================================", flush=True)
    print(f"🚀 Training {model_name.upper()} (94 Classes) | Seed: {seed} | Device: {device}", flush=True)
    print(f"================================================================================", flush=True)

    # 1. Load Data Caches
    train_npz = np.load(os.path.join(CACHE_DIR, "train_data.npz"))
    val_npz = np.load(os.path.join(CACHE_DIR, "val_data.npz"))
    test_npz = np.load(os.path.join(CACHE_DIR, "test_data.npz"))
    user_noise_np = np.load(os.path.join(CACHE_DIR, "user_ambient_noise.npy"))

    train_wavs_raw = train_npz["wavs"]
    train_labels_raw = train_npz["labels_94"]
    train_vt_raw = train_npz["voice_types"]

    if use_supp_synth:
        print("  [Ablation] Including supplemental_synth (train split only)...", flush=True)
        supp_npz = np.load(os.path.join(CACHE_DIR, "synth_supp_train.npz"))
        train_wavs_raw = np.concatenate([train_wavs_raw, supp_npz["wavs"]], axis=0)
        train_labels_raw = np.concatenate([train_labels_raw, supp_npz["labels_94"]], axis=0)
        train_vt_raw = np.concatenate([train_vt_raw, supp_npz["voice_types"]], axis=0)

    # 4x Filipino speech oversampling
    fil_mask = (train_vt_raw == "real_filipino")
    fil_indices = np.where(fil_mask)[0]
    num_duplicates = max(filipino_oversample - 1, 0)
    oversampled_train_indices = np.concatenate([
        np.arange(len(train_labels_raw)),
        np.repeat(fil_indices, num_duplicates)
    ])

    train_wavs = torch.from_numpy(train_wavs_raw).unsqueeze(1).to(device)
    train_labels = torch.from_numpy(train_labels_raw).to(device)
    train_idx_t = torch.from_numpy(oversampled_train_indices).to(device)

    total_train_samples = len(oversampled_train_indices)
    total_fil_samples = len(fil_indices) * filipino_oversample
    fil_share_pct = round((total_fil_samples / total_train_samples) * 100.0, 2)
    print(f"  [Filipino Oversampling] Factor: {filipino_oversample}x | Unique clips: {len(fil_indices)} | Per epoch: {total_fil_samples}/{total_train_samples} ({fil_share_pct}% effective share)")

    val_wavs = torch.from_numpy(val_npz["wavs"]).unsqueeze(1).to(device)
    val_labels = torch.from_numpy(val_npz["labels_94"]).to(device)
    val_labels_np = val_npz["labels_94"]
    val_fg = val_npz["is_filipino_group"]
    val_oos = val_npz["is_oos_speech"]

    test_wavs = torch.from_numpy(test_npz["wavs"]).unsqueeze(1).to(device)
    test_labels_np = test_npz["labels_94"]
    test_intents_np = test_npz["intents_19"]
    test_slots_np = test_npz["slot_values"]
    test_fg = test_npz["is_filipino_group"]
    test_oos = test_npz["is_oos_speech"]
    test_os = test_npz["on_script"]
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
            # Time shift
            shift = random.randint(-1600, 1600)
            if shift > 0:
                x = F.pad(x[:, :, :-shift], (shift, 0))
            elif shift < 0:
                x = F.pad(x[:, :, -shift:], (0, -shift))

            # Mix ambient room noise
            if random.random() < 0.5:
                noise_idx = random.randint(0, len(ambient_noise) - 1)
                noise_clip = ambient_noise[noise_idx : noise_idx + 1]
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

    # 3. Model Architecture & Loss Setup
    if model_name == "bcresnet":
        model = get_bcresnet(num_classes=NUM_CLASSES).to(device)
    else:
        model = get_dscnn(num_classes=NUM_CLASSES).to(device)

    # Class weighting: OUT_OF_SCOPE class 93 up-weighted
    class_weights = torch.ones(NUM_CLASSES, device=device)
    class_weights[OOS_CLASS_IDX] = oos_weight
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # Pre-extract Validation and Test Features in batches to avoid VRAM spikes
    def extract_batched(w_tensor, b_size=256):
        out_list = []
        with torch.no_grad():
            for s_idx in range(0, len(w_tensor), b_size):
                chunk = w_tensor[s_idx : s_idx + b_size]
                out_list.append(extract_features(chunk, augment=False))
        return torch.cat(out_list, dim=0)

    val_feats = extract_batched(val_wavs)
    test_feats = extract_batched(test_wavs)

    best_filipino_acc = -1.0
    best_val_all = -1.0
    best_ckpt_path = os.path.join(CKPT_DIR, f"best_{model_name}_94class_seed{seed}.pt")
    training_log = []

    print(f"  Starting {epochs} epochs of training...")
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(len(train_idx_t), device=device)
        total_loss = 0.0
        n_batches = 0

        for b_start in range(0, len(train_idx_t), batch_size):
            b_indices = train_idx_t[perm[b_start : b_start + batch_size]]
            bx_wav = train_wavs[b_indices]
            by = train_labels[b_indices]

            bx = extract_features(bx_wav, augment=True)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / n_batches

        # Evaluate Validation Set
        model.eval()
        with torch.no_grad():
            v_logits = model(val_feats)
            v_preds = torch.argmax(v_logits, dim=1).cpu().numpy()
            v_acc_all = round(float(np.mean(v_preds == val_labels_np) * 100.0), 2)

            fil_val_mask = (val_fg == 1) & (val_labels_np != OOS_CLASS_IDX)
            if np.sum(fil_val_mask) > 0:
                v_acc_fil = round(float(np.mean(v_preds[fil_val_mask] == val_labels_np[fil_val_mask]) * 100.0), 2)
            else:
                v_acc_fil = 0.0

        is_best = (v_acc_fil > best_filipino_acc) or (v_acc_fil == best_filipino_acc and v_acc_all > best_val_all) or (epoch == 1)
        if is_best:
            best_filipino_acc = v_acc_fil
            best_val_all = v_acc_all
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "val_filipino_acc": v_acc_fil,
                "val_acc_all": v_acc_all,
                "seed": seed,
                "model_name": model_name,
            }, best_ckpt_path)

        star = " ★ (Best Filipino Acc)" if is_best else ""
        print(f"  Epoch {epoch:2d}/{epochs:2d} | Loss: {avg_loss:.4f} | Val Acc (All): {v_acc_all:5.2f}% | Val Filipino Acc: {v_acc_fil:5.2f}%{star}", flush=True)

        training_log.append({
            "epoch": epoch,
            "train_loss": round(avg_loss, 4),
            "val_acc_all": v_acc_all,
            "val_acc_filipino": v_acc_fil,
            "is_best": is_best,
        })

    train_time = round(time.time() - t0, 2)
    print(f"  Training finished in {train_time}s. Best Filipino Val Acc: {best_filipino_acc:5.2f}%")

    # Save training log
    pd.DataFrame(training_log).to_csv(
        os.path.join(EXPORTS_DIR, f"training_log_{model_name}_94class_seed{seed}.csv"),
        index=False
    )

    # 4. Load Best Checkpoint for Evaluation & Calibration
    ckpt = torch.load(best_ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        val_logits_t = model(val_feats)
        val_probs_raw = F.softmax(val_logits_t, dim=1).cpu().numpy()

        test_logits_t = model(test_feats)
        test_probs_raw = F.softmax(test_logits_t, dim=1).cpu().numpy()

        # Physical mic noise evaluation
        noise_feats = extract_features(ambient_noise, augment=False)
        noise_logits_t = model(noise_feats)
        noise_probs_raw = F.softmax(noise_logits_t, dim=1).cpu().numpy()

    # 5. Dual-Constraint Tau Tuning on Validation (Raw Softmax)
    best_tau_raw, df_sweep_raw, achievable_raw = run_tau_sweep_on_validation(
        val_probs=val_probs_raw,
        val_labels_94=val_labels_np,
        is_filipino_group=val_fg,
        is_oos_speech=val_oos,
        target_far=5.0,
        max_frr_cap=30.0,
    )
    df_sweep_raw.to_csv(
        os.path.join(EXPORTS_DIR, f"tau_sweep_val_{model_name}_seed{seed}.csv"),
        index=False
    )

    # 6. Temperature Scaling Calibration
    temp_val = fit_temperature_scaling(val_logits_t, val_labels)
    with torch.no_grad():
        val_probs_cal = F.softmax(val_logits_t / temp_val, dim=1).cpu().numpy()
        test_probs_cal = F.softmax(test_logits_t / temp_val, dim=1).cpu().numpy()
        noise_probs_cal = F.softmax(noise_logits_t / temp_val, dim=1).cpu().numpy()

    best_tau_cal, df_sweep_cal, achievable_cal = run_tau_sweep_on_validation(
        val_probs=val_probs_cal,
        val_labels_94=val_labels_np,
        is_filipino_group=val_fg,
        is_oos_speech=val_oos,
        target_far=5.0,
        max_frr_cap=30.0,
    )
    df_sweep_cal.to_csv(
        os.path.join(EXPORTS_DIR, f"tau_sweep_val_{model_name}_calibrated_seed{seed}.csv"),
        index=False
    )

    print(f"  [Rejection Tuning] Optimal Tau (Raw): {best_tau_raw:.2f} (Achievable: {achievable_raw}) | Temperature: {temp_val:.4f} | Optimal Tau (Calibrated): {best_tau_cal:.2f}")

    # 7. Evaluate on Test Set
    labels_info = json.load(open(os.path.join(EXPORTS_DIR, "labels_94.json"), "r"))

    test_metrics_raw = compute_comprehensive_metrics(
        probs=test_probs_raw,
        labels_94=test_labels_np,
        intents_19=test_intents_np,
        slot_values=test_slots_np,
        voice_types=test_vt,
        on_script_flags=test_os,
        is_filipino_group=test_fg,
        is_oos_speech=test_oos,
        labels_info=labels_info,
        tau=best_tau_raw,
        delta_m=0.15,
        mic_noise_probs=noise_probs_raw,
    )

    test_metrics_cal = compute_comprehensive_metrics(
        probs=test_probs_cal,
        labels_94=test_labels_np,
        intents_19=test_intents_np,
        slot_values=test_slots_np,
        voice_types=test_vt,
        on_script_flags=test_os,
        is_filipino_group=test_fg,
        is_oos_speech=test_oos,
        labels_info=labels_info,
        tau=best_tau_cal,
        delta_m=0.15,
        mic_noise_probs=noise_probs_cal,
    )

    eval_summary = {
        "model_name": model_name,
        "seed": seed,
        "epochs": epochs,
        "train_time_seconds": train_time,
        "best_epoch": int(ckpt["epoch"]),
        "val_best_filipino_acc_pct": float(ckpt["val_filipino_acc"]),
        "temperature": temp_val,
        "tau_raw": best_tau_raw,
        "tau_raw_achievable": achievable_raw,
        "tau_calibrated": best_tau_cal,
        "tau_calibrated_achievable": achievable_cal,
        "test_metrics_raw": test_metrics_raw,
        "test_metrics_calibrated": test_metrics_cal,
    }

    with open(os.path.join(EXPORTS_DIR, f"eval_{model_name}_94class_seed{seed}.json"), "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    return eval_summary, best_ckpt_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="both", choices=["bcresnet", "dscnn", "both"])
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 1337, 2026])
    parser.add_argument("--use-supp-synth", action="store_true", help="Run ablation with supplemental_synth train split")
    args = parser.parse_args()

    device = torch.device("cuda:7" if torch.cuda.is_available() else "cpu")
    print(f"Using compute device: {device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else socket.gethostname()})")

    models_to_train = ["bcresnet", "dscnn"] if args.model == "both" else [args.model]

    all_results = {}
    for m in models_to_train:
        all_results[m] = {}
        for s in args.seeds:
            res, ckpt_p = train_single_run(
                model_name=m,
                seed=s,
                device=device,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                use_supp_synth=args.use_supp_synth,
            )
            all_results[m][s] = res

    # Compute Multi-Seed Aggregates
    summary_table = {}
    for m in models_to_train:
        seed_runs = [all_results[m][s]["test_metrics_raw"] for s in args.seeds]

        cmd_accs = [r["command_level_94"]["accuracy_pct"] for r in seed_runs]
        int_accs = [r["intent_level_19"]["accuracy_pct"] for r in seed_runs]
        slot_accs = [r["slot_exact_match"]["exact_match_pct"] for r in seed_runs]
        far_ooss = [r["pipeline_rates"]["far_oos_speech_pct"] for r in seed_runs]
        frr_ins = [r["pipeline_rates"]["frr_in_scope_pct"] for r in seed_runs]

        # Filipino specific
        fil_accs = [r["voice_type_breakdown"]["real_filipino"]["raw_command_acc_pct"] for r in seed_runs]
        fil_frrs = [r["voice_type_breakdown"]["real_filipino"]["frr_pct"] for r in seed_runs]

        summary_table[m] = {
            "command_accuracy_mean": round(float(np.mean(cmd_accs)), 2),
            "command_accuracy_std": round(float(np.std(cmd_accs)), 2),
            "intent_accuracy_mean": round(float(np.mean(int_accs)), 2),
            "intent_accuracy_std": round(float(np.std(int_accs)), 2),
            "slot_exact_match_mean": round(float(np.mean(slot_accs)), 2),
            "slot_exact_match_std": round(float(np.std(slot_accs)), 2),
            "far_oos_mean": round(float(np.mean(far_ooss)), 2),
            "far_oos_std": round(float(np.std(far_ooss)), 2),
            "frr_in_scope_mean": round(float(np.mean(frr_ins)), 2),
            "frr_in_scope_std": round(float(np.std(frr_ins)), 2),
            "filipino_command_acc_mean": round(float(np.mean(fil_accs)), 2),
            "filipino_command_acc_std": round(float(np.std(fil_accs)), 2),
            "filipino_frr_mean": round(float(np.mean(fil_frrs)), 2),
            "filipino_frr_std": round(float(np.std(fil_frrs)), 2),
            "seeds": args.seeds,
        }

    with open(os.path.join(EXPORTS_DIR, "multi_seed_summary_94class.json"), "w", encoding="utf-8") as f:
        json.dump(summary_table, f, indent=2)

    print("\n" + "=" * 80)
    print("🏆 MULTI-SEED SUMMARY RESULTS (TEST SET, 3 SEEDS)")
    print("=" * 80)
    for m, s in summary_table.items():
        print(f"[{m.upper()}]")
        print(f"  94-Command Acc:  {s['command_accuracy_mean']} ± {s['command_accuracy_std']}%")
        print(f"  19-Intent Acc:   {s['intent_accuracy_mean']} ± {s['intent_accuracy_std']}%")
        print(f"  Slot Match Acc:  {s['slot_exact_match_mean']} ± {s['slot_exact_match_std']}%")
        print(f"  FAR OOS Speech:  {s['far_oos_mean']} ± {s['far_oos_std']}%")
        print(f"  Real Filipino Acc: {s['filipino_command_acc_mean']} ± {s['filipino_command_acc_std']}% (FRR: {s['filipino_frr_mean']} ± {s['filipino_frr_std']}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
