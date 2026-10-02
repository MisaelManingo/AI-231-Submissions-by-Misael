#!/usr/bin/env python3
"""
export_94class_onnx.py

Exports trained 94-class models (BC-ResNet-1 and DS-CNN) to ONNX:
- FP32 ONNX export with dynamic batch axes
- INT8 dynamic quantization
- Verifies FP32 vs INT8 accuracy on unseen test set (CPU onnxruntime) at BOTH:
    - 94-command level
    - 19-intent level
- Evaluates FLOPs, MACs, parameter counts, and file sizes using vcmbench/flops.py (onnx_profile)
- Saves comprehensive report to exports/v3_94class/export_report_94class.json
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torchaudio
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from sklearn.metrics import accuracy_score, f1_score

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.insert(0, os.path.expanduser("~/vcm-benchmark"))

from models.bcresnet import get_bcresnet
from models.dscnn import get_dscnn
from vcmbench.flops import onnx_profile, format_si

CACHE_DIR = os.path.join(BASE_DIR, "data", "v3_cache_94class")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports", "v3_94class")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "v3_94class")
os.makedirs(EXPORTS_DIR, exist_ok=True)

TARGET_SR = 16000
TARGET_SAMPLES = 32000
NUM_CLASSES = 94


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def extract_features_cpu(wavs_np, batch_size=128):
    """Extract log-mel spectrogram features matching pure NumPy exactly."""
    melspec = torchaudio.transforms.MelSpectrogram(
        sample_rate=TARGET_SR, n_fft=400, win_length=400, hop_length=160, n_mels=40
    )
    a2db = torchaudio.transforms.AmplitudeToDB()

    features = []
    wavs_tensor = torch.from_numpy(wavs_np).unsqueeze(1)
    with torch.no_grad():
        for i in range(0, len(wavs_tensor), batch_size):
            bx = wavs_tensor[i : i + batch_size]
            mel = melspec(bx)
            db = a2db(mel)
            features.append(db.numpy())
    return np.concatenate(features, axis=0)


def evaluate_onnx_session(session, features, labels_94, intents_19, var_to_intent_idx, batch_size=128):
    input_name = session.get_inputs()[0].name
    preds_94 = []
    for i in range(0, len(features), batch_size):
        bx = features[i : i + batch_size]
        outputs = session.run(None, {input_name: bx})[0]
        preds_94.extend(np.argmax(outputs, axis=1))

    preds_94 = np.array(preds_94)
    preds_int = var_to_intent_idx[preds_94]

    acc_94 = round(float(accuracy_score(labels_94, preds_94) * 100.0), 2)
    f1_94 = round(float(f1_score(labels_94, preds_94, average="macro", zero_division=0)), 4)
    acc_int = round(float(accuracy_score(intents_19, preds_int) * 100.0), 2)
    f1_int = round(float(f1_score(intents_19, preds_int, average="macro", zero_division=0)), 4)
    return acc_94, f1_94, acc_int, f1_int


def export_and_quantize(model_name, ckpt_path, test_features, test_labels_94, test_intents_19, var_to_intent_idx):
    print(f"\n=======================================================")
    print(f"📦 Exporting {model_name.upper()} (94 Classes) to ONNX")
    print(f"=======================================================")
    print(f"  Checkpoint: {ckpt_path}")

    # 1. Instantiate Model & Load Weights
    if model_name == "bcresnet":
        model = get_bcresnet(num_classes=NUM_CLASSES)
    elif model_name == "dscnn":
        model = get_dscnn(num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model name {model_name}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    param_count = count_parameters(model)
    param_count_m = round(param_count / 1e6, 4)
    print(f"  Parameters: {param_count:,} ({param_count_m} M)")

    # 2. Export FP32 ONNX
    fp32_path = os.path.join(EXPORTS_DIR, f"{model_name}_94class_fp32.onnx")
    dummy_input = torch.randn(1, 1, 40, 201, dtype=torch.float32)

    torch.onnx.export(
        model,
        dummy_input,
        fp32_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False,
    )
    onnx_model = onnx.load(fp32_path)
    onnx.checker.check_model(onnx_model)
    fp32_size_mb = round(os.path.getsize(fp32_path) / (1024 * 1024), 3)
    print(f"  ✓ FP32 ONNX Exported: {fp32_path} ({fp32_size_mb} MB)")

    # 3. Dynamic INT8 Quantization
    int8_path = os.path.join(EXPORTS_DIR, f"{model_name}_94class_int8.onnx")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
    )
    int8_size_mb = round(os.path.getsize(int8_path) / (1024 * 1024), 3)
    compression_ratio = round(fp32_size_mb / int8_size_mb, 2)
    print(f"  ✓ INT8 ONNX Quantized: {int8_path} ({int8_size_mb} MB, {compression_ratio}x compression)")

    # 4. Profile with vcmbench/flops.py
    fp32_profile = onnx_profile(fp32_path)
    int8_profile = onnx_profile(int8_path)
    print(f"  [vcmbench flops] FP32 MACs: {format_si(fp32_profile['macs'])}, FLOPs: {format_si(fp32_profile['flops'])}")
    print(f"  [vcmbench flops] INT8 MACs: {format_si(int8_profile['macs'])}, FLOPs: {format_si(int8_profile['flops'])}")

    # 5. Evaluate CPU onnxruntime Accuracy
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.inter_op_num_threads = 1

    fp32_sess = ort.InferenceSession(fp32_path, sess_opts, providers=["CPUExecutionProvider"])
    int8_sess = ort.InferenceSession(int8_path, sess_opts, providers=["CPUExecutionProvider"])

    fp32_94_acc, fp32_94_f1, fp32_int_acc, fp32_int_f1 = evaluate_onnx_session(
        fp32_sess, test_features, test_labels_94, test_intents_19, var_to_intent_idx
    )
    int8_94_acc, int8_94_f1, int8_int_acc, int8_int_f1 = evaluate_onnx_session(
        int8_sess, test_features, test_labels_94, test_intents_19, var_to_intent_idx
    )

    drop_94 = round(fp32_94_acc - int8_94_acc, 2)
    drop_int = round(fp32_int_acc - int8_int_acc, 2)

    print(f"  FP32 Test: 94-Command Acc={fp32_94_acc}%, 19-Intent Acc={fp32_int_acc}%")
    print(f"  INT8 Test: 94-Command Acc={int8_94_acc}%, 19-Intent Acc={int8_int_acc}%")
    print(f"  Accuracy Drop (FP32 -> INT8): 94-Command: {drop_94:+.2f}%, 19-Intent: {drop_int:+.2f}%")

    return {
        "model_name": model_name,
        "checkpoint": ckpt_path,
        "parameters": param_count,
        "parameters_m": param_count_m,
        "fp32": {
            "path": fp32_path,
            "size_mb": fp32_size_mb,
            "macs": fp32_profile["macs"],
            "macs_si": format_si(fp32_profile["macs"]),
            "flops": fp32_profile["flops"],
            "flops_si": format_si(fp32_profile["flops"]),
            "command_acc_94": fp32_94_acc,
            "intent_acc_19": fp32_int_acc,
        },
        "int8": {
            "path": int8_path,
            "size_mb": int8_size_mb,
            "compression_ratio": compression_ratio,
            "macs": int8_profile["macs"],
            "macs_si": format_si(int8_profile["macs"]),
            "flops": int8_profile["flops"],
            "flops_si": format_si(int8_profile["flops"]),
            "command_acc_94": int8_94_acc,
            "intent_acc_19": int8_int_acc,
            "drop_command_94_pct": drop_94,
            "drop_intent_19_pct": drop_int,
        },
    }


def main():
    print("=" * 65)
    print("🚀 EXPORT & QUANTIZE 94-CLASS MODELS TO ONNX")
    print("=" * 65)

    test_npz = np.load(os.path.join(CACHE_DIR, "test_data.npz"))
    test_wavs = test_npz["wavs"]
    test_labels_94 = test_npz["labels_94"]
    test_intents_19 = test_npz["intents_19"]

    labels_info = json.load(open(os.path.join(EXPORTS_DIR, "labels_94.json"), "r"))
    var_to_intent_idx = np.array([labels_info["var_to_intent_idx"][str(i)] for i in range(NUM_CLASSES)])

    print("Extracting CPU log-mel features for test split...")
    test_features = extract_features_cpu(test_wavs)

    report = {}
    for m in ["bcresnet", "dscnn"]:
        ckpt_p = os.path.join(CKPT_DIR, f"best_{m}_94class_seed42.pt")
        if os.path.exists(ckpt_p):
            res = export_and_quantize(m, ckpt_p, test_features, test_labels_94, test_intents_19, var_to_intent_idx)
            report[m] = res
        else:
            print(f"Warning: Checkpoint {ckpt_p} not found, skipping {m}")

    with open(os.path.join(EXPORTS_DIR, "export_report_94class.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved export report to {os.path.join(EXPORTS_DIR, 'export_report_94class.json')}")


if __name__ == "__main__":
    main()
