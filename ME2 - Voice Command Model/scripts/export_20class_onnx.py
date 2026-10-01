#!/usr/bin/env python3
"""
export_20class_onnx.py

Exports the trained 20-class PyTorch models (BC-ResNet-1 and DS-CNN) to ONNX:
- FP32 ONNX export with dynamic batch axes
- INT8 dynamic quantization
- Verifies FP32 vs INT8 accuracy on unseen test set (CPU onnxruntime)
- Measures model parameter counts (M) and disk file sizes (MB)
- Saves export report to exports/v2_20class/export_report_20class.json
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

from models.bcresnet import get_bcresnet
from models.dscnn import get_dscnn

CACHE_DIR = os.path.join(BASE_DIR, "data", "v2_cache_20class")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports", "v2_20class")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints", "v2")
os.makedirs(EXPORTS_DIR, exist_ok=True)

TARGET_SR = 16000
TARGET_SAMPLES = 32000
NUM_CLASSES = 20


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def extract_features_cpu(wavs_np, batch_size=128):
    """Extract log-mel spectrogram features using torchaudio on CPU."""
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
            mean = db.mean(dim=(-2, -1), keepdim=True)
            std = db.std(dim=(-2, -1), keepdim=True) + 1e-5
            normed = (db - mean) / std
            features.append(normed.numpy())
    return np.concatenate(features, axis=0)


def evaluate_onnx_session(session, features, labels, batch_size=128):
    input_name = session.get_inputs()[0].name
    preds = []
    for i in range(0, len(features), batch_size):
        bx = features[i : i + batch_size]
        outputs = session.run(None, {input_name: bx})[0]
        preds.extend(np.argmax(outputs, axis=1))

    preds = np.array(preds)
    acc = accuracy_score(labels, preds) * 100.0
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    return round(float(acc), 2), round(float(f1), 4)


def export_and_quantize(model_name, ckpt_path, test_features, test_labels):
    print(f"\n=======================================================")
    print(f"📦 Exporting {model_name.upper()} (20 Classes) to ONNX")
    print(f"=======================================================")
    print(f"  Checkpoint: {ckpt_path}")

    # 1. Instantiate PyTorch Model & Load Checkpoint
    if model_name == "bcresnet":
        model = get_bcresnet(num_classes=NUM_CLASSES)
    elif model_name == "dscnn":
        model = get_dscnn(num_classes=NUM_CLASSES)
    else:
        raise ValueError(f"Unknown model name {model_name}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    param_count = count_parameters(model)
    param_count_m = round(param_count / 1e6, 4)
    print(f"  Parameters: {param_count:,} ({param_count_m} M)")

    # 2. Export FP32 ONNX
    fp32_path = os.path.join(EXPORTS_DIR, f"{model_name}_20class_fp32.onnx")
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
    int8_path = os.path.join(EXPORTS_DIR, f"{model_name}_20class_int8.onnx")
    quantize_dynamic(
        model_input=fp32_path,
        model_output=int8_path,
        weight_type=QuantType.QInt8,
    )
    int8_size_mb = round(os.path.getsize(int8_path) / (1024 * 1024), 3)
    compression_ratio = round(fp32_size_mb / int8_size_mb, 2)
    print(f"  ✓ INT8 ONNX Quantized: {int8_path} ({int8_size_mb} MB, {compression_ratio}x compression)")

    # 4. Evaluate Test Accuracy Drop on CPU onnxruntime
    print(f"  Evaluating FP32 and INT8 models on unseen test split (CPU onnxruntime)...")
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 4
    sess_opts.inter_op_num_threads = 1

    fp32_sess = ort.InferenceSession(fp32_path, sess_opts, providers=["CPUExecutionProvider"])
    int8_sess = ort.InferenceSession(int8_path, sess_opts, providers=["CPUExecutionProvider"])

    fp32_acc, fp32_f1 = evaluate_onnx_session(fp32_sess, test_features, test_labels)
    int8_acc, int8_f1 = evaluate_onnx_session(int8_sess, test_features, test_labels)
    acc_drop = round(fp32_acc - int8_acc, 2)

    print(f"  FP32 Test Acc: {fp32_acc}% | Macro F1: {fp32_f1}")
    print(f"  INT8 Test Acc: {int8_acc}% | Macro F1: {int8_f1}")
    print(f"  Accuracy Drop (FP32 -> INT8): {acc_drop:+.2f}%")

    return {
        "model_name": model_name,
        "checkpoint": ckpt_path,
        "parameters": param_count,
        "parameters_m": param_count_m,
        "fp32": {
            "path": fp32_path,
            "size_mb": fp32_size_mb,
            "test_accuracy": fp32_acc,
            "test_macro_f1": fp32_f1,
        },
        "int8": {
            "path": int8_path,
            "size_mb": int8_size_mb,
            "compression_ratio": compression_ratio,
            "test_accuracy": int8_acc,
            "test_macro_f1": int8_f1,
            "accuracy_drop_pct": acc_drop,
        },
    }


def main():
    print("=" * 60)
    print("🚀 EXPORT & QUANTIZE 20-CLASS MODELS TO ONNX")
    print("=" * 60)

    # Load test split for verification
    test_npz = np.load(os.path.join(CACHE_DIR, "test_data.npz"))
    test_wavs = test_npz["wavs"]
    test_labels = test_npz["labels"]
    print(f"Loaded {len(test_wavs)} test waveforms from cache. Extracting features on CPU...")
    test_features = extract_features_cpu(test_wavs)
    print(f"Features ready: shape {test_features.shape}")

    models = ["bcresnet", "dscnn"]
    export_report = {}

    for m_name in models:
        ckpt_path = os.path.join(CKPT_DIR, f"{m_name}_20class_final.pt")
        if not os.path.exists(ckpt_path):
            raise FileNotFoundError(f"Final checkpoint not found: {ckpt_path}. Run train_20class.py first.")
        report = export_and_quantize(m_name, ckpt_path, test_features, test_labels)
        export_report[m_name] = report

    report_path = os.path.join(EXPORTS_DIR, "export_report_20class.json")
    with open(report_path, "w") as f:
        json.dump(export_report, f, indent=2)

    print("\n" + "=" * 60)
    print(f"✅ Export completed successfully! Summary saved to {report_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
