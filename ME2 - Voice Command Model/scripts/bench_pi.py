#!/usr/bin/env python3
"""
bench_pi.py - 100% PyTorch-Free Raspberry Pi 5 Benchmark Script

Measures and prints:
- Keyword Accuracy (%)
- Intent Accuracy (%)
- False Accept Rate (FAR) (%) at decision threshold 0.65
- Latency p50 (ms), Latency p95 (ms), Latency Mean (ms)
- Real-Time Factor (RTF)
- System Hardware: Raspberry Pi Model, OS / Kernel, ONNX Runtime version, Thread Count

Requirements on Raspberry Pi:
  pip install numpy onnxruntime
(Zero PyTorch, zero Librosa dependencies)
"""

import os
import sys
import time
import json
import platform
import argparse
import numpy as np
import onnxruntime as ort

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXPORTS_DIR = os.path.join(BASE_DIR, "exports", "v2")
CACHE_DIR = os.path.join(BASE_DIR, "data", "v2_cache")

TARGET_SR = 16000
TARGET_DURATION = 2.0
TARGET_SAMPLES = 32000
DECISION_THRESHOLD = 0.65


class PureNumpyFeatureExtractor:
    """
    100% PyTorch-free audio feature extractor matching torch.transforms.MelSpectrogram
    and AmplitudeToDB with standard Mel-scale and Hann windowing.
    """
    def __init__(self, mel_filters_path, hann_window_path):
        if not os.path.exists(mel_filters_path):
            # Fallback to parent exports dir if not in v2
            fallback = os.path.join(BASE_DIR, "exports", "mel_filters_40.npy")
            mel_filters_path = fallback if os.path.exists(fallback) else mel_filters_path
        if not os.path.exists(hann_window_path):
            fallback = os.path.join(BASE_DIR, "exports", "hann_window_400.npy")
            hann_window_path = fallback if os.path.exists(fallback) else hann_window_path

        self.mel_filters = np.load(mel_filters_path)  # (40, 201)
        self.hann_window = np.load(hann_window_path)  # (400,)
        self.n_fft = 400
        self.hop_length = 160

    def compute_melspectrogram(self, audio):
        """
        Computes 40-mel log-mel spectrogram matching Torchaudio.
        audio: 1D array of 32,000 float32 samples.
        returns: (1, 1, 40, 201) float32 array
        """
        if audio.ndim > 1:
            audio = np.mean(audio, axis=0) if audio.shape[0] < audio.shape[1] else np.mean(audio, axis=1)

        if len(audio) < TARGET_SAMPLES:
            pad = TARGET_SAMPLES - len(audio)
            audio = np.pad(audio, (pad // 2, pad - pad // 2), mode="constant")
        elif len(audio) > TARGET_SAMPLES:
            start = (len(audio) - TARGET_SAMPLES) // 2
            audio = audio[start : start + TARGET_SAMPLES]

        pad_amount = self.n_fft // 2
        y_padded = np.pad(audio, pad_amount, mode="reflect")

        num_frames = (len(y_padded) - self.n_fft) // self.hop_length + 1
        strides = (self.hop_length * y_padded.strides[0], y_padded.strides[0])
        frames = np.lib.stride_tricks.as_strided(y_padded, shape=(num_frames, self.n_fft), strides=strides)

        stft = np.fft.rfft(frames * self.hann_window, n=self.n_fft, axis=1)
        pow_np = (np.abs(stft) ** 2).T  # (201, num_frames)
        mel = np.dot(self.mel_filters.T, pow_np)  # (40, num_frames)

        amin = 1e-10
        log_spec = 10.0 * np.log10(np.maximum(amin, mel))
        mean = np.mean(log_spec)
        std = np.std(log_spec) + 1e-5
        normed = (log_spec - mean) / std

        return normed[np.newaxis, np.newaxis, :, :].astype(np.float32)


def get_hardware_info():
    """Detects Raspberry Pi model and operating system."""
    model_name = "Unknown"
    # Raspberry Pi device tree model path
    if os.path.exists("/proc/device-tree/model"):
        try:
            with open("/proc/device-tree/model", "r") as f:
                model_name = f.read().strip().replace("\x00", "")
        except Exception:
            pass

    if model_name == "Unknown":
        model_name = f"{platform.system()} {platform.machine()}"

    os_info = platform.platform()
    if os.path.exists("/etc/os-release"):
        try:
            with open("/etc/os-release", "r") as f:
                for line in f:
                    if line.startswith("PRETTY_NAME="):
                        os_info = line.split("=", 1)[1].strip().strip('"')
                        break
        except Exception:
            pass

    return {
        "hardware_model": model_name,
        "os_version": os_info,
        "kernel": platform.release(),
        "arch": platform.machine(),
        "python_version": platform.python_version(),
    }


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)


def run_benchmark(model_path, threads, num_warmup, num_runs, data_split="holdout"):
    print("=" * 70)
    print("🍓 RASPBERRY PI 5 VOICE COMMAND MODEL BENCHMARK")
    print("=" * 70)

    hw = get_hardware_info()
    print(f"  Device Model:   {hw['hardware_model']}")
    print(f"  OS / Kernel:    {hw['os_version']} ({hw['kernel']})")
    print(f"  Architecture:   {hw['arch']} | Python: {hw['python_version']}")
    print(f"  ONNX Runtime:   v{ort.__version__}")
    print(f"  Model File:     {model_path}")
    print(f"  Thread Count:   {threads}")
    print(f"  Warmup Runs:    {num_warmup} | Timed Runs: {num_runs}")

    # Set ONNX Runtime Session Options
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    # 1. Initialize Feature Extractor
    mel_path = os.path.join(EXPORTS_DIR, "mel_filters_40.npy")
    hann_path = os.path.join(EXPORTS_DIR, "hann_window_400.npy")
    fe = PureNumpyFeatureExtractor(mel_path, hann_path)

    # 2. Check for Validation Dataset
    labels_file = os.path.join(EXPORTS_DIR, "labels_32.json")
    labels_meta = None
    if os.path.exists(labels_file):
        with open(labels_file, "r") as f:
            labels_meta = json.load(f)

    data_file = os.path.join(CACHE_DIR, f"{data_split}_data.npz")
    has_dataset = os.path.exists(data_file) and labels_meta is not None

    kw_acc = "<TBD>"
    it_acc = "<TBD>"
    far_val = "<TBD>"

    if has_dataset:
        print(f"\n[1/3] Evaluating Accuracy & FAR on '{data_split}' split ({data_file})...")
        npz = np.load(data_file)
        wavs = npz["wavs"]
        labels = npz["labels"]

        intent_map = labels_meta["intent_map"]
        idx2label = {int(k): v for k, v in labels_meta["idx2label"].items()}
        unique_intents = sorted(list(set(intent_map.values())))
        intent2idx = {it: i for i, it in enumerate(unique_intents)}
        class_to_intent = np.array([intent2idx[intent_map[idx2label[c]]] for c in range(len(idx2label))])

        correct_kw = 0
        correct_it = 0
        false_accepts = 0
        bg_samples = 0

        for idx in range(len(labels)):
            feat = fe.compute_melspectrogram(wavs[idx])
            logits = session.run([output_name], {input_name: feat})[0][0]
            probs = softmax(logits)
            pred = int(np.argmax(probs))
            conf = float(probs[pred])
            true_lbl = int(labels[idx])

            if pred == true_lbl:
                correct_kw += 1
            if class_to_intent[pred] == class_to_intent[true_lbl]:
                correct_it += 1

            if true_lbl == 0:
                bg_samples += 1
                if pred != 0 and conf >= DECISION_THRESHOLD:
                    false_accepts += 1

        kw_acc = round((correct_kw / len(labels)) * 100.0, 2)
        it_acc = round((correct_it / len(labels)) * 100.0, 2)
        far_val = round((false_accepts / max(1, bg_samples)) * 100.0, 2) if bg_samples > 0 else 0.0

        print(f"      Keyword Accuracy: {kw_acc}% ({correct_kw}/{len(labels)})")
        print(f"      Intent Accuracy:  {it_acc}% ({correct_it}/{len(labels)})")
        print(f"      False Accept Rate: {far_val}% (Threshold: {DECISION_THRESHOLD})")
    else:
        print(f"\n[1/3] Dataset file '{data_file}' not found. Accuracy marked as <TBD>.")

    # 3. Latency & Real-Time Factor (RTF) Benchmarking
    print(f"\n[2/3] Performing {num_warmup} warmup runs...")
    dummy_wav = np.random.randn(TARGET_SAMPLES).astype(np.float32) * 0.1
    dummy_feat = fe.compute_melspectrogram(dummy_wav)

    for _ in range(num_warmup):
        _ = session.run([output_name], {input_name: dummy_feat})

    print(f"[3/3] Performing {num_runs} timed inference passes...")
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = session.run([output_name], {input_name: dummy_feat})
        t1 = time.perf_counter()
        latencies.append((t1 - t0) * 1000.0)  # ms

    latencies = np.array(latencies)
    lat_mean = round(float(np.mean(latencies)), 2)
    lat_p50 = round(float(np.percentile(latencies, 50)), 2)
    lat_p95 = round(float(np.percentile(latencies, 95)), 2)
    lat_p99 = round(float(np.percentile(latencies, 99)), 2)
    lat_min = round(float(np.min(latencies)), 2)
    lat_max = round(float(np.max(latencies)), 2)

    # Real-Time Factor: inference time / input audio duration (2.0s)
    rtf = round(float((lat_p95 / 1000.0) / TARGET_DURATION), 5)

    print("\n" + "=" * 70)
    print("⏱️ LATENCY BENCHMARK RESULTS:")
    print("=" * 70)
    print(f"  Inference Latency (Mean): {lat_mean:6.2f} ms")
    print(f"  Inference Latency (p50):  {lat_p50:6.2f} ms")
    print(f"  Inference Latency (p95):  {lat_p95:6.2f} ms")
    print(f"  Inference Latency (p99):  {lat_p99:6.2f} ms")
    print(f"  Inference Latency (Min):  {lat_min:6.2f} ms | Max: {lat_max:6.2f} ms")
    print(f"  Real-Time Factor (RTF):   {rtf:.5f} (p95 / 2.0s)")

    # 4. Formatted Markdown Output for README
    print("\n" + "=" * 70)
    print("📋 COPY & PASTE FOR README.md:")
    print("=" * 70)
    md_output = f"""
## Validation on the Raspberry Pi
- Keyword / intent acc: {kw_acc}% / {it_acc}%
- False-accept rate: {far_val}%
- Latency p95 / RTF: {lat_p95} ms / {rtf}
- Latency p50: {lat_p50} ms
- Runtime: onnxruntime v{ort.__version__} · {threads} thr ({hw['hardware_model']})
"""
    print(md_output.strip())
    print("=" * 70)

    # Save to JSON
    result = {
        "hardware": hw,
        "onnxruntime_version": ort.__version__,
        "threads": threads,
        "model_file": os.path.basename(model_path),
        "keyword_accuracy": kw_acc,
        "intent_accuracy": it_acc,
        "false_accept_rate": far_val,
        "latency_ms": {
            "mean": lat_mean,
            "p50": lat_p50,
            "p95": lat_p95,
            "p99": lat_p99,
            "min": lat_min,
            "max": lat_max,
        },
        "rtf_p95": rtf,
    }

    out_file = os.path.join(EXPORTS_DIR, "pi_benchmark_results.json")
    with open(out_file, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved benchmark results to {out_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 Model Benchmark")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(EXPORTS_DIR, "bcresnet_v2_int8.onnx"),
        help="Path to INT8 ONNX model",
    )
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads to use")
    parser.add_argument("--warmup", type=int, default=30, help="Number of warmup inference runs")
    parser.add_argument("--runs", type=int, default=200, help="Number of timed inference runs (at least 200)")
    parser.add_argument(
        "--split",
        type=str,
        default="holdout",
        choices=["holdout", "test"],
        help="Dataset split to evaluate accuracy",
    )
    args = parser.parse_args()

    run_benchmark(
        model_path=args.model,
        threads=args.threads,
        num_warmup=args.warmup,
        num_runs=args.runs,
        data_split=args.split,
    )


if __name__ == "__main__":
    main()
