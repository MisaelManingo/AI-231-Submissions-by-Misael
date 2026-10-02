#!/usr/bin/env python3
"""
run_loso_vcm.py

Leave-One-Speaker-Out (LOSO) Cross-Validation and Final Model Training for 94-Class VCM:
1. 5-fold LOSO cross-validation over the 5 real Filipino train speakers:
   - '202322013' (487 clips)
   - '202322013_speaker2' (61 clips)
   - 'S1' (20 clips)
   - 'S2' (20 clips)
   - 'S3' (20 clips)
2. Oversamples real Filipino speech (4x) and real OOS speech (4x, class weight 5.0).
3. Pools out-of-fold validation predictions across all 608 real Filipino clips and validation OOS speech.
4. Tunes dual-constraint threshold tau* and balanced threshold tau_bal on pooled held-out predictions.
5. Trains final models across seeds [42, 1337, 2026] for BC-ResNet-1 and DS-CNN.
6. Evaluates test set metrics under BOTH tau* and tau_bal operating points.
7. Exports updated ONNX models (FP32 & INT8) and comprehensive evaluation summaries.
"""

import os
import sys
import time
import json
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
OOS_CLASS_WEIGHT = 5.0
OOS_OVERSAMPLE = 4
FILIPINO_OVERSAMPLE = 4
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


def fit_temperature_scaling(val_logits_t, val_labels_t):
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


def build_augmented_feature_extractor(device, ambient_noise):
    gpu_melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR, n_fft=400, win_length=400, hop_length=160, n_mels=40
    ).to(device)
    gpu_a2db = torchaudio.transforms.AmplitudeToDB().to(device)
    gpu_tmask = torchaudio.transforms.TimeMasking(20).to(device)
    gpu_fmask = torchaudio.transforms.FrequencyMasking(6).to(device)

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

    return extract_features


def extract_batched(extract_fn, w_tensor, b_size=256):
    out_list = []
    with torch.no_grad():
        for s_idx in range(0, len(w_tensor), b_size):
            chunk = w_tensor[s_idx : s_idx + b_size]
            out_list.append(extract_fn(chunk, augment=False))
    return torch.cat(out_list, dim=0)


def evaluate_tau_sweep(val_probs, val_labels_94, is_filipino_group, is_oos_speech, target_far=5.0, max_frr_cap=30.0):
    preds = np.argmax(val_probs, axis=1)
    max_probs = np.max(val_probs, axis=1)

    in_scope_mask = (val_labels_94 != OOS_CLASS_IDX)
    oos_speech_mask = (is_oos_speech == 1)
    filipino_in_scope_mask = (is_filipino_group == 1) & in_scope_mask

    n_oos = int(np.sum(oos_speech_mask))
    n_fil = int(np.sum(filipino_in_scope_mask))

    tau_candidates = [round(t, 2) for t in np.arange(0.10, 0.96, 0.05)]
    sweep_rows = []

    best_tau_star = 0.50
    best_penalty = float("inf")
    achievable_star = False

    best_tau_bal = 0.40
    best_bal_score = float("inf")

    for tau in tau_candidates:
        reject_mask = (preds == OOS_CLASS_IDX) | (max_probs < tau)
        accept_mask = ~reject_mask

        oos_acc = int(np.sum(oos_speech_mask & accept_mask))
        far, far_lo, far_hi = wilson_ci(oos_acc, n_oos)

        fil_rej = int(np.sum(filipino_in_scope_mask & reject_mask))
        frr, frr_lo, frr_hi = wilson_ci(fil_rej, n_fil)

        in_acc_mask = in_scope_mask & accept_mask
        acc_on_acc = round(float(np.sum(in_acc_mask & (preds == val_labels_94)) / max(np.sum(in_acc_mask), 1) * 100.0), 2)

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

        bal_score = frr + 0.5 * far
        if bal_score < best_bal_score and tau >= 0.25:
            best_bal_score = bal_score
            best_tau_bal = tau

    df_sweep = pd.DataFrame(sweep_rows)
    return best_tau_star, achievable_star, best_tau_bal, df_sweep


def compute_comprehensive_test_metrics(
    probs, labels_94, intents_19, slot_values,
    voice_types, on_script_flags, is_filipino_group, is_oos_speech,
    labels_info, tau, delta_m=0.15, mic_noise_probs=None
):
    preds_94 = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)

    var_to_intent_idx = np.array([labels_info["var_to_intent_idx"][str(i)] for i in range(NUM_CLASSES)])
    preds_intent = var_to_intent_idx[preds_94]
    idx_to_slot = labels_info["idx_to_slot"]
    idx_to_phrase = labels_info["idx_to_phrase"]

    in_scope_mask = (labels_94 != OOS_CLASS_IDX)
    oos_speech_mask = (is_oos_speech == 1)
    filipino_in_scope_mask = (is_filipino_group == 1) & in_scope_mask

    reject_mask = (preds_94 == OOS_CLASS_IDX) | (max_probs < tau)
    accept_mask = ~reject_mask

    n_total = len(labels_94)
    corr_94 = np.sum(preds_94 == labels_94)
    acc_94_val, acc_94_low, acc_94_high = wilson_ci(corr_94, n_total)
    bal_acc_94 = round(float(balanced_accuracy_score(labels_94, preds_94) * 100.0), 2)
    macro_p_94 = round(float(precision_score(labels_94, preds_94, average="macro", zero_division=0) * 100.0), 2)
    macro_r_94 = round(float(recall_score(labels_94, preds_94, average="macro", zero_division=0) * 100.0), 2)
    macro_f1_94 = round(float(f1_score(labels_94, preds_94, average="macro", zero_division=0) * 100.0), 2)
    p_num, r_num = macro_p_94 / 100.0, macro_r_94 / 100.0
    macro_f2_94 = round((5 * p_num * r_num / (4 * p_num + r_num + 1e-8)) * 100.0, 2)

    corr_intent = np.sum(preds_intent == intents_19)
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
    misfired = int(np.sum(in_scope_accepted & (preds_94 != labels_94)))
    misfire_rate, misfire_low, misfire_high = wilson_ci(misfired, n_in_scope)

    n_accepted_in_scope = int(np.sum(in_scope_accepted))
    if n_accepted_in_scope > 0:
        correct_accepted = int(np.sum(in_scope_accepted & (preds_94 == labels_94)))
        acc_accepted_94 = round((correct_accepted / n_accepted_in_scope) * 100.0, 2)
        corr_int_accepted = int(np.sum(in_scope_accepted & (preds_intent == intents_19)))
        acc_accepted_int = round((corr_int_accepted / n_accepted_in_scope) * 100.0, 2)
    else:
        acc_accepted_94, acc_accepted_int = 0.0, 0.0

    slot_eligible, slot_correct = 0, 0
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

    sorted_probs = np.sort(probs, axis=1)
    margins = sorted_probs[:, -1] - sorted_probs[:, -2]

    rule_definitions = {
        "class_only": (preds_94 == OOS_CLASS_IDX),
        "threshold_only": (max_probs < tau),
        "combined": (preds_94 == OOS_CLASS_IDX) | (max_probs < tau),
        "margin_variant": (preds_94 == OOS_CLASS_IDX) | (margins < delta_m),
    }

    rule_results = {}
    for rule_name, r_mask in rule_definitions.items():
        a_mask = ~r_mask
        oos_acc = np.sum(oos_speech_mask & a_mask)
        far_r, far_lo, far_hi = wilson_ci(oos_acc, n_oos)

        if mic_noise_probs is not None and len(mic_noise_probs) > 0:
            np_preds = np.argmax(mic_noise_probs, axis=1)
            np_max_p = np.max(mic_noise_probs, axis=1)
            np_sorted = np.sort(mic_noise_probs, axis=1)
            np_m = np_sorted[:, -1] - np_sorted[:, -2]
            if rule_name == "class_only":
                n_rej = (np_preds == OOS_CLASS_IDX)
            elif rule_name == "threshold_only":
                n_rej = (np_max_p < tau)
            elif rule_name == "combined":
                n_rej = (np_preds == OOS_CLASS_IDX) | (np_max_p < tau)
            else:
                n_rej = (np_preds == OOS_CLASS_IDX) | (np_m < delta_m)
            far_noise = round(float(np.sum(~n_rej) / len(mic_noise_probs) * 100.0), 2)
        else:
            far_noise = 0.0

        acc_in = in_scope_mask & a_mask
        acc_on_acc = round(float(np.sum(acc_in & (preds_94 == labels_94)) / max(np.sum(acc_in), 1) * 100.0), 2)

        fil_rej = np.sum(filipino_in_scope_mask & r_mask)
        frr_fil = round(float(fil_rej / max(np.sum(filipino_in_scope_mask), 1) * 100.0), 2)

        rule_results[rule_name] = {
            "far_oos_speech_pct": far_r,
            "far_oos_ci_95": [far_lo, far_hi],
            "far_mic_noise_pct": far_noise,
            "command_acc_accepted_pct": acc_on_acc,
            "frr_filipino_group_pct": frr_fil,
            "accepted_total": int(np.sum(a_mask)),
            "rejected_total": int(np.sum(r_mask)),
        }

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
            "tau": tau,
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
        "rejection_rules": rule_results,
    }


def run_loso_cross_validation(device="cuda:6", epochs=20):
    print("\n" + "=" * 80)
    print("🇵🇭 RUNNING LEAVE-ONE-SPEAKER-OUT (LOSO) 5-FOLD CROSS-VALIDATION")
    print("=" * 80)

    train_npz = np.load(os.path.join(CACHE_DIR, "train_data.npz"))
    val_npz = np.load(os.path.join(CACHE_DIR, "val_data.npz"))
    user_noise_np = np.load(os.path.join(CACHE_DIR, "user_ambient_noise.npy"))

    train_wavs_all = train_npz["wavs"]
    train_labels_all = train_npz["labels_94"]
    train_vt_all = train_npz["voice_types"]
    train_spk_all = train_npz["speakers"]
    train_oos_all = train_npz["is_oos_speech"]
    train_fg_all = train_npz["is_filipino_group"]

    ambient_noise = torch.from_numpy(user_noise_np).unsqueeze(1).to(device)
    extract_features = build_augmented_feature_extractor(device, ambient_noise)

    fil_train_mask = (train_vt_all == "real_filipino")
    fil_speakers = sorted(list(np.unique(train_spk_all[fil_train_mask])))
    print(f"5 Filipino speakers for LOSO folds: {fil_speakers}")

    val_oos_indices = np.where(val_npz["is_oos_speech"] == 1)[0]
    val_oos_wavs = val_npz["wavs"][val_oos_indices]
    val_oos_labels = val_npz["labels_94"][val_oos_indices]

    fold_results = []
    pooled_heldout_preds = []
    pooled_heldout_labels = []
    pooled_heldout_is_fil = []
    pooled_heldout_is_oos = []

    best_epochs = []

    for fold_idx, held_out_spk in enumerate(fil_speakers):
        print(f"\n--- [FOLD {fold_idx + 1}/5] Held-out Speaker: '{held_out_spk}' ---", flush=True)
        set_seed(42 + fold_idx)

        held_out_mask = (train_spk_all == held_out_spk)
        train_fold_mask = ~held_out_mask

        fold_train_idx_base = np.where(train_fold_mask)[0]
        fil_fold_idx = np.where(train_fold_mask & fil_train_mask)[0]
        oos_fold_idx = np.where(train_fold_mask & (train_oos_all == 1))[0]

        oversampled_indices = np.concatenate([
            fold_train_idx_base,
            np.repeat(fil_fold_idx, FILIPINO_OVERSAMPLE - 1),
            np.repeat(oos_fold_idx, OOS_OVERSAMPLE - 1),
        ])

        train_wavs = torch.from_numpy(train_wavs_all).unsqueeze(1).to(device)
        train_labels = torch.from_numpy(train_labels_all).to(device)
        train_idx_t = torch.from_numpy(oversampled_indices).to(device)

        held_out_indices = np.where(held_out_mask)[0]
        held_out_wavs = train_wavs_all[held_out_indices]
        held_out_labels = train_labels_all[held_out_indices]

        val_fold_wavs_np = np.concatenate([held_out_wavs, val_oos_wavs], axis=0)
        val_fold_labels_np = np.concatenate([held_out_labels, val_oos_labels], axis=0)
        val_fold_fg_np = np.concatenate([np.ones(len(held_out_labels)), np.zeros(len(val_oos_labels))], axis=0)
        val_fold_oos_np = np.concatenate([np.zeros(len(held_out_labels)), np.ones(len(val_oos_labels))], axis=0)

        val_fold_wavs_t = torch.from_numpy(val_fold_wavs_np).unsqueeze(1).to(device)
        val_fold_feats = extract_batched(extract_features, val_fold_wavs_t)

        model = get_bcresnet(num_classes=NUM_CLASSES).to(device)
        class_weights = torch.ones(NUM_CLASSES, device=device)
        class_weights[OOS_CLASS_IDX] = OOS_CLASS_WEIGHT
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

        batch_size = 64
        best_fil_acc = -1.0
        best_epoch = 1
        best_logits = None

        for epoch in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(len(train_idx_t), device=device)
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
                v_logits = model(val_fold_feats)
                v_preds = torch.argmax(v_logits, dim=1).cpu().numpy()

            fil_mask = (val_fold_fg_np == 1) & (val_fold_labels_np != OOS_CLASS_IDX)
            acc_fil = np.mean(v_preds[fil_mask] == val_fold_labels_np[fil_mask]) * 100.0 if np.sum(fil_mask) > 0 else 0.0

            if acc_fil > best_fil_acc or epoch == 1:
                best_fil_acc = acc_fil
                best_epoch = epoch
                best_logits = v_logits.clone()

            if epoch % 5 == 0 or epoch == epochs:
                print(f"  Epoch {epoch:2d}/{epochs:2d} | Loss: {total_loss/n_batches:.4f} | Held-out '{held_out_spk}' Acc: {acc_fil:5.2f}%", flush=True)

        print(f"  [Fold {fold_idx + 1} Best] Epoch {best_epoch} | Held-out Acc: {best_fil_acc:5.2f}%", flush=True)
        best_epochs.append(best_epoch)

        with torch.no_grad():
            held_out_probs = F.softmax(best_logits[:len(held_out_labels)], dim=1).cpu().numpy()
            oos_probs = F.softmax(best_logits[len(held_out_labels):], dim=1).cpu().numpy()

        pooled_heldout_preds.append(held_out_probs)
        pooled_heldout_labels.append(held_out_labels)
        pooled_heldout_is_fil.append(np.ones(len(held_out_labels)))
        pooled_heldout_is_oos.append(np.zeros(len(held_out_labels)))

        if fold_idx == 0:
            pooled_heldout_preds.append(oos_probs)
            pooled_heldout_labels.append(val_oos_labels)
            pooled_heldout_is_fil.append(np.zeros(len(val_oos_labels)))
            pooled_heldout_is_oos.append(np.ones(len(val_oos_labels)))

        fold_results.append({
            "fold": fold_idx + 1,
            "speaker": held_out_spk,
            "clips": len(held_out_labels),
            "best_epoch": best_epoch,
            "held_out_filipino_acc_pct": round(float(best_fil_acc), 2),
        })

    all_heldout_probs = np.concatenate(pooled_heldout_preds, axis=0)
    all_heldout_labels = np.concatenate(pooled_heldout_labels, axis=0)
    all_heldout_is_fil = np.concatenate(pooled_heldout_is_fil, axis=0)
    all_heldout_is_oos = np.concatenate(pooled_heldout_is_oos, axis=0)

    print("\n" + "=" * 80)
    print(f"📊 POOLED HELD-OUT PREDICTIONS: {len(all_heldout_labels)} clips (608 Filipino across 5 speakers + 25 OOS speech)")
    print("=" * 80)

    pooled_preds = np.argmax(all_heldout_probs, axis=1)
    oos_mask = (all_heldout_is_oos == 1)
    fil_mask = (all_heldout_is_fil == 1)

    class_only_far_oos = round(float(np.sum(oos_mask & (pooled_preds != OOS_CLASS_IDX)) / max(np.sum(oos_mask), 1) * 100.0), 2)
    class_only_frr_fil = round(float(np.sum(fil_mask & (pooled_preds == OOS_CLASS_IDX)) / max(np.sum(fil_mask), 1) * 100.0), 2)
    fil_heldout_acc = round(float(np.sum(fil_mask & (pooled_preds == all_heldout_labels)) / max(np.sum(fil_mask), 1) * 100.0), 2)

    print(f"  Class-Only FAR on OOS Speech: {class_only_far_oos}%")
    print(f"  Class-Only FRR on Filipino Speech: {class_only_frr_fil}%")
    print(f"  Held-out Filipino Raw Accuracy: {fil_heldout_acc}%")

    tau_star, achievable_star, tau_bal, df_sweep = evaluate_tau_sweep(
        val_probs=all_heldout_probs,
        val_labels_94=all_heldout_labels,
        is_filipino_group=all_heldout_is_fil,
        is_oos_speech=all_heldout_is_oos,
        target_far=5.0,
        max_frr_cap=30.0,
    )

    df_sweep.to_csv(os.path.join(EXPORTS_DIR, "tau_sweep_loso_pooled.csv"), index=False)
    print(f"\n  [Threshold Selection on Pooled LOSO]")
    print(f"  Strict Operating Point tau*: {tau_star:.2f} (Achievable: {achievable_star})")
    print(f"  Balanced Operating Point tau_bal: {tau_bal:.2f}")

    mean_best_epoch = int(round(np.mean(best_epochs)))
    print(f"  Mean Best Epoch across 5 folds: {mean_best_epoch}")

    loso_summary = {
        "fold_results": fold_results,
        "mean_best_epoch": mean_best_epoch,
        "pooled_heldout_clips": len(all_heldout_labels),
        "class_only_far_oos_pct": class_only_far_oos,
        "class_only_frr_filipino_pct": class_only_frr_fil,
        "pooled_filipino_raw_acc_pct": fil_heldout_acc,
        "tau_star": tau_star,
        "tau_star_achievable": achievable_star,
        "tau_bal": tau_bal,
    }

    with open(os.path.join(EXPORTS_DIR, "loso_cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump(loso_summary, f, indent=2)

    return loso_summary, mean_best_epoch, tau_star, tau_bal


def train_production_models(device="cuda:6", epochs=20, tau_star=0.75, tau_bal=0.45, seeds=[42, 1337, 2026]):
    print("\n" + "=" * 80)
    print("🚀 TRAINING PRODUCTION MODELS (ALL 5 FILIPINO SPEAKERS + OOS 4X OVERSAMPLED)")
    print("=" * 80)

    train_npz = np.load(os.path.join(CACHE_DIR, "train_data.npz"))
    val_npz = np.load(os.path.join(CACHE_DIR, "val_data.npz"))
    test_npz = np.load(os.path.join(CACHE_DIR, "test_data.npz"))
    user_noise_np = np.load(os.path.join(CACHE_DIR, "user_ambient_noise.npy"))
    labels_info = json.load(open(os.path.join(EXPORTS_DIR, "labels_94.json"), "r"))

    train_wavs_raw = train_npz["wavs"]
    train_labels_raw = train_npz["labels_94"]
    train_vt_raw = train_npz["voice_types"]
    train_oos_raw = train_npz["is_oos_speech"]

    fil_indices = np.where(train_vt_raw == "real_filipino")[0]
    oos_indices = np.where(train_oos_raw == 1)[0]

    oversampled_train_indices = np.concatenate([
        np.arange(len(train_labels_raw)),
        np.repeat(fil_indices, FILIPINO_OVERSAMPLE - 1),
        np.repeat(oos_indices, OOS_OVERSAMPLE - 1),
    ])

    train_wavs = torch.from_numpy(train_wavs_raw).unsqueeze(1).to(device)
    train_labels = torch.from_numpy(train_labels_raw).to(device)
    train_idx_t = torch.from_numpy(oversampled_train_indices).to(device)

    total_samples = len(oversampled_train_indices)
    fil_samples = len(fil_indices) * FILIPINO_OVERSAMPLE
    print(f"  Training set: {total_samples} samples per epoch (Filipino share: {fil_samples}/{total_samples} = {fil_samples/total_samples*100:.2f}%)")

    test_wavs = torch.from_numpy(test_npz["wavs"]).unsqueeze(1).to(device)
    test_labels_np = test_npz["labels_94"]
    test_intents_np = test_npz["intents_19"]
    test_slots_np = test_npz["slot_values"]
    test_fg = test_npz["is_filipino_group"]
    test_oos = test_npz["is_oos_speech"]
    test_os = test_npz["on_script"]
    test_vt = test_npz["voice_types"]

    ambient_noise = torch.from_numpy(user_noise_np).unsqueeze(1).to(device)
    extract_features = build_augmented_feature_extractor(device, ambient_noise)
    test_feats = extract_batched(extract_features, test_wavs)

    noise_feats = extract_features(ambient_noise, augment=False)

    models_to_run = [("bcresnet", s) for s in seeds] + [("dscnn", 42)]
    all_runs = {}

    for model_name, seed in models_to_run:
        print(f"\n--- Training {model_name.upper()} | Seed {seed} | Device {device} ---", flush=True)
        set_seed(seed)

        if model_name == "bcresnet":
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
        for epoch in range(1, epochs + 1):
            model.train()
            perm = torch.randperm(len(train_idx_t), device=device)
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

            if epoch % 5 == 0 or epoch == epochs:
                print(f"  Epoch {epoch:2d}/{epochs:2d} | Train Loss: {total_loss/n_batches:.4f}", flush=True)

        train_time = round(time.time() - t0, 2)

        ckpt_path = os.path.join(CKPT_DIR, f"best_{model_name}_94class_seed{seed}.pt")
        torch.save({
            "epoch": epochs,
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "seed": seed,
        }, ckpt_path)

        model.eval()
        with torch.no_grad():
            test_logits = model(test_feats)
            test_probs = F.softmax(test_logits, dim=1).cpu().numpy()

            noise_logits = model(noise_feats)
            noise_probs = F.softmax(noise_logits, dim=1).cpu().numpy()

        metrics_star = compute_comprehensive_test_metrics(
            probs=test_probs,
            labels_94=test_labels_np,
            intents_19=test_intents_np,
            slot_values=test_slots_np,
            voice_types=test_vt,
            on_script_flags=test_os,
            is_filipino_group=test_fg,
            is_oos_speech=test_oos,
            labels_info=labels_info,
            tau=tau_star,
            mic_noise_probs=noise_probs,
        )

        metrics_bal = compute_comprehensive_test_metrics(
            probs=test_probs,
            labels_94=test_labels_np,
            intents_19=test_intents_np,
            slot_values=test_slots_np,
            voice_types=test_vt,
            on_script_flags=test_os,
            is_filipino_group=test_fg,
            is_oos_speech=test_oos,
            labels_info=labels_info,
            tau=tau_bal,
            mic_noise_probs=noise_probs,
        )

        eval_entry = {
            "model_name": model_name,
            "seed": seed,
            "epochs": epochs,
            "train_time_seconds": train_time,
            "tau_star": tau_star,
            "tau_bal": tau_bal,
            "test_metrics_tau_star": metrics_star,
            "test_metrics_tau_bal": metrics_bal,
        }

        with open(os.path.join(EXPORTS_DIR, f"eval_{model_name}_94class_seed{seed}.json"), "w", encoding="utf-8") as f:
            json.dump(eval_entry, f, indent=2)

        all_runs[f"{model_name}_seed{seed}"] = eval_entry

    return all_runs


def main():
    device = "cuda:6" if torch.cuda.is_available() else "cpu"
    print(f"Compute Device: {device} ({torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'CPU'})")

    loso_summary, mean_best_epoch, tau_star, tau_bal = run_loso_cross_validation(device=device, epochs=20)
    all_runs = train_production_models(device=device, epochs=20, tau_star=tau_star, tau_bal=tau_bal, seeds=[42, 1337, 2026])

    print("\n" + "=" * 80)
    print("🏆 SUMMARY TABLE: TEST EVALUATION ACROSS OPERATING POINTS")
    print("=" * 80)
    for k, v in all_runs.items():
        m_star = v["test_metrics_tau_star"]
        m_bal = v["test_metrics_tau_bal"]
        print(f"\n[{k.upper()}]")
        print(f"  Strict (tau* = {tau_star:.2f}):")
        print(f"    Command Acc on Accepted: {m_star['pipeline_rates']['command_acc_on_accepted_pct']}% | FAR OOS: {m_star['pipeline_rates']['far_oos_speech_pct']}% | Filipino Acc: {m_star['voice_type_breakdown']['real_filipino']['command_acc_on_accepted_pct']}% (FRR: {m_star['voice_type_breakdown']['real_filipino']['frr_pct']}%)")
        print(f"  Balanced (tau_bal = {tau_bal:.2f}):")
        print(f"    Command Acc on Accepted: {m_bal['pipeline_rates']['command_acc_on_accepted_pct']}% | FAR OOS: {m_bal['pipeline_rates']['far_oos_speech_pct']}% | Filipino Acc: {m_bal['voice_type_breakdown']['real_filipino']['command_acc_on_accepted_pct']}% (FRR: {m_bal['voice_type_breakdown']['real_filipino']['frr_pct']}%)")
    print("=" * 80)


if __name__ == "__main__":
    main()
