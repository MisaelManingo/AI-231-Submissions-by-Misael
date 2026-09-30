import os
import json
import argparse
import time
import torch
import numpy as np
import onnx
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from dataset import VoiceCommandDataset
from models.dscnn import get_dscnn
from models.bcresnet import get_bcresnet

BASE_DIR = "/home/misael.andre.maningo/MEng AI/AI 231/ME2 - Voice Command Model"
OPTIONB_DIR = os.path.join(BASE_DIR, "upstream_repo/MEX2/OptionB")
MANIFEST_PATH = os.path.join(BASE_DIR, "data/unified_manifest.csv")
LABELS_PATH = os.path.join(BASE_DIR, "data/labels_20.json")

def parse_args():
    parser = argparse.ArgumentParser(description="Export and Quantize VCM to ONNX")
    parser.add_argument("--model", type=str, default="dscnn", choices=["dscnn", "bcresnet"], help="Model architecture")
    return parser.parse_args()


def benchmark_onnx(onnx_path, dummy_input, num_warmup=100, num_runs=500):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    # Warmup
    for _ in range(num_warmup):
        _ = session.run(None, {input_name: dummy_input})
        
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = session.run(None, {input_name: dummy_input})
        latencies.append((time.perf_counter() - t0) * 1000.0) # ms
        
    latencies = np.array(latencies)
    return {
        "mean_ms": np.mean(latencies),
        "p50_ms": np.percentile(latencies, 50),
        "p95_ms": np.percentile(latencies, 95),
        "p99_ms": np.percentile(latencies, 99),
        "min_ms": np.min(latencies),
        "max_ms": np.max(latencies)
    }


def evaluate_onnx(onnx_path, dataset):
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, num_workers=2)
    all_preds = []
    all_targets = []
    
    for inputs, targets in loader:
        ort_inputs = {input_name: inputs.numpy()}
        outputs = session.run(None, ort_inputs)[0]
        preds = np.argmax(outputs, axis=1)
        all_preds.extend(preds)
        all_targets.extend(targets.numpy())
        
    acc = accuracy_score(all_targets, all_preds)
    f1 = f1_score(all_targets, all_preds, average="macro", zero_division=0)
    return acc, f1


def main():
    args = parse_args()
    print("=" * 60)
    print(f"Exporting & Quantizing Model: {args.model.upper()}")
    print("=" * 60)
    
    # Load label map
    with open(LABELS_PATH, "r") as f:
        label_info = json.load(f)
    num_classes = len(label_info["label2idx"])
    
    # Load PyTorch checkpoint
    ckpt_path = os.path.join(BASE_DIR, "checkpoints", f"best_{args.model}.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
        
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if args.model == "dscnn":
        model = get_dscnn(num_classes=num_classes)
    elif args.model == "bcresnet":
        model = get_bcresnet(num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    # 1. Export PyTorch to FP32 ONNX
    fp32_onnx_path = os.path.join(BASE_DIR, "exports", f"{args.model}_fp32.onnx")
    dummy_input = torch.randn(1, 1, 40, 201, dtype=torch.float32)
    
    torch.onnx.export(
        model,
        dummy_input,
        fp32_onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch_size"}, "logits": {0: "batch_size"}},
        dynamo=False
    )
    print(f"Exported FP32 ONNX model to: {fp32_onnx_path}")
    onnx_model = onnx.load(fp32_onnx_path)
    onnx.checker.check_model(onnx_model)
    print("FP32 ONNX model verified successfully.")
    
    # File size
    fp32_size_mb = os.path.getsize(fp32_onnx_path) / (1024 * 1024)
    print(f"FP32 Model Size: {fp32_size_mb:.2f} MB")
    
    # 2. Dynamic INT8 Quantization
    int8_onnx_path = os.path.join(BASE_DIR, "exports", f"{args.model}_int8.onnx")
    quantize_dynamic(
        model_input=fp32_onnx_path,
        model_output=int8_onnx_path,
        weight_type=QuantType.QInt8
    )
    print(f"Quantized INT8 ONNX model to: {int8_onnx_path}")
    int8_size_mb = os.path.getsize(int8_onnx_path) / (1024 * 1024)
    print(f"INT8 Model Size: {int8_size_mb:.2f} MB (Compression: {fp32_size_mb / int8_size_mb:.2f}x)")
    
    # 3. CPU Benchmarking
    print("\n--- CPU Latency Benchmark (1 Sample = 2.0s audio) ---")
    dummy_np = dummy_input.numpy()
    
    fp32_bench = benchmark_onnx(fp32_onnx_path, dummy_np)
    print(f"FP32 CPU Latency -> Mean: {fp32_bench['mean_ms']:.2f}ms | p50: {fp32_bench['p50_ms']:.2f}ms | p95: {fp32_bench['p95_ms']:.2f}ms | p99: {fp32_bench['p99_ms']:.2f}ms")
    
    int8_bench = benchmark_onnx(int8_onnx_path, dummy_np)
    print(f"INT8 CPU Latency -> Mean: {int8_bench['mean_ms']:.2f}ms | p50: {int8_bench['p50_ms']:.2f}ms | p95: {int8_bench['p95_ms']:.2f}ms | p99: {int8_bench['p99_ms']:.2f}ms")
    
    # 4. Accuracy Verification on Test Split
    print("\n--- Verifying Test Accuracy After Quantization ---")
    df = pd.read_csv(MANIFEST_PATH)
    test_df = df[df["split"] == "test"]
    test_ds = VoiceCommandDataset(test_df, OPTIONB_DIR, is_train=False)
    
    fp32_acc, fp32_f1 = evaluate_onnx(fp32_onnx_path, test_ds)
    int8_acc, int8_f1 = evaluate_onnx(int8_onnx_path, test_ds)
    
    print(f"FP32 Test Accuracy: {fp32_acc*100:.2f}% | Macro F1: {fp32_f1:.4f}")
    print(f"INT8 Test Accuracy: {int8_acc*100:.2f}% | Macro F1: {int8_f1:.4f}")
    print(f"Accuracy delta from quantization: {(int8_acc - fp32_acc)*100:+.2f}%")
    
    # Save export report
    report_path = os.path.join(BASE_DIR, "exports", f"export_report_{args.model}.json")
    with open(report_path, "w") as f:
        json.dump({
            "model": args.model,
            "fp32_size_mb": fp32_size_mb,
            "int8_size_mb": int8_size_mb,
            "compression_ratio": fp32_size_mb / int8_size_mb,
            "fp32_latency_ms": fp32_bench,
            "int8_latency_ms": int8_bench,
            "fp32_test_acc": fp32_acc,
            "fp32_test_f1": fp32_f1,
            "int8_test_acc": int8_acc,
            "int8_test_f1": int8_f1
        }, f, indent=2)
    print(f"\nExport benchmark report saved to: {report_path}")

if __name__ == "__main__":
    main()
