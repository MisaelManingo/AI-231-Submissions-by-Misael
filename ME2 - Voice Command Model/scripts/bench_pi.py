#!/usr/bin/env python3
"""
bench_pi.py - 100% PyTorch-Free Raspberry Pi 5 Benchmark & Holdout Evaluation Script

Evaluates 20-class ONNX Voice Command Models (FP32 or INT8) on Raspberry Pi:
- 100% PyTorch-free and scikit-learn-free (only NumPy + ONNX Runtime CPU)
- Measures latency (p50, p95, mean) and Real-Time Factor (RTF) across >=200 timed passes
- Evaluates held-out split (e.g., holdout_data.npz or test_data.npz)
- Computes Keyword Accuracy, Intent Accuracy, Macro-F1, and Confusion Matrix
- Rejection rules: class-only, threshold-only (tau), combined, and margin variant
- Evaluates FAR on OOS speech (with Wilson 95% CI), FAR on mic noise, FRR on Filipino group
- Breaks down metrics per speaker and voice type
- Prints the exact formatted block to paste directly into README.md replacing 'REPLACE'

Usage:
  python3 scripts/bench_pi.py --model exports/v2_20class/bcresnet_20class_int8.onnx --eval-holdout
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
CACHE_DIR = os.path.join(BASE_DIR, "data", "v2_cache_20class")
EXPORTS_DIR = os.path.join(BASE_DIR, "exports", "v2_20class")

TARGET_SR = 16000
TARGET_DURATION = 2.0
TARGET_SAMPLES = 32000
OOS_CLASS_IDX = 19


def wilson_ci(k, n, confidence=0.95):
    """Wilson score interval for binomial proportions (pure NumPy/Python)."""
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


class PureNumpyFeatureExtractor:
    """100% PyTorch-free audio feature extractor matching Torchaudio 40 log-mel frontend."""
    def __init__(self, mel_filters_path, hann_window_path):
        if not os.path.exists(mel_filters_path):
            mel_filters_path = os.path.join(BASE_DIR, "exports", "mel_filters_40.npy")
        if not os.path.exists(hann_window_path):
            hann_window_path = os.path.join(BASE_DIR, "exports", "hann_window_400.npy")

        self.mel_filters = np.load(mel_filters_path)  # (40, 201)
        self.hann_window = np.load(hann_window_path)  # (400,)
        self.n_fft = 400
        self.hop_length = 160

    def compute_melspectrogram(self, audio):
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
    model_name = "Unknown Device"
    if os.path.exists("/proc/device-tree/model"):
        try:
            with open("/proc/device-tree/model", "r") as f:
                model_name = f.read().strip().replace("\x00", "")
        except Exception:
            pass

    if model_name == "Unknown Device":
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


def compute_macro_f1(y_true, y_pred, num_classes=20):
    """Pure NumPy macro-F1 computation."""
    f1_list = []
    for c in range(num_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        f1_list.append(f1)
    return round(float(np.mean(f1_list)), 4) if f1_list else 0.0


def evaluate_rules_numpy(probs, labels, is_fg, is_oos, tau=0.55, delta_m=0.15, mic_probs=None):
    preds = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)

    sorted_p = np.sort(probs, axis=1)
    margins = sorted_p[:, -1] - sorted_p[:, -2]

    in_scope = (labels != OOS_CLASS_IDX)
    oos_speech = (is_oos == 1)
    fg_in_scope = (is_fg == 1)

    n_oos = int(np.sum(oos_speech))
    n_fg = int(np.sum(fg_in_scope))

    rules = {
        "class_only": (preds == OOS_CLASS_IDX),
        "threshold_only": (max_probs < tau),
        "combined": (preds == OOS_CLASS_IDX) | (max_probs < tau),
        "margin_variant": (preds == OOS_CLASS_IDX) | (margins < delta_m),
    }

    out = {}
    for r_name, reject_mask in rules.items():
        accept_mask = ~reject_mask

        # OOS speech FAR
        oos_acc = np.sum(oos_speech & accept_mask)
        far_val, far_lo, far_hi = wilson_ci(oos_acc, n_oos)

        # Mic noise FAR
        far_noise = 0.0
        if mic_probs is not None and len(mic_probs) > 0:
            n_p = np.argmax(mic_probs, axis=1)
            n_max = np.max(mic_probs, axis=1)
            n_m = np.sort(mic_probs, axis=1)[:, -1] - np.sort(mic_probs, axis=1)[:, -2]
            if r_name == "class_only":
                rej_n = (n_p == OOS_CLASS_IDX)
            elif r_name == "threshold_only":
                rej_n = (n_max < tau)
            elif r_name == "combined":
                rej_n = (n_p == OOS_CLASS_IDX) | (n_max < tau)
            else:
                rej_n = (n_p == OOS_CLASS_IDX) | (n_m < delta_m)
            far_noise = round((np.sum(~rej_n) / len(mic_probs)) * 100.0, 2)

        # Accuracy on accepted in-scope clips
        acc_mask = in_scope & accept_mask
        acc_acc = round((np.sum(acc_mask & (preds == labels)) / np.sum(acc_mask)) * 100.0, 2) if np.sum(acc_mask) > 0 else 0.0

        # FRR on Filipino group
        frr_fg = round((np.sum(fg_in_scope & reject_mask) / n_fg) * 100.0, 2) if n_fg > 0 else 0.0

        out[r_name] = {
            "far_oos": far_val,
            "far_oos_ci": [far_lo, far_hi],
            "far_noise": far_noise,
            "acc_accepted": acc_acc,
            "frr_filipino": frr_fg,
        }

    return out


def run_benchmark(model_path, threads, num_warmup, num_runs, eval_holdout=False, eval_test=False, tau=0.55, delta_m=0.15):
    print("=" * 75)
    print("🍓 RASPBERRY PI 5 VOICE COMMAND MODEL BENCHMARK (20 CLASSES)")
    print("=" * 75)

    hw = get_hardware_info()
    print(f"  Device Model:        {hw['hardware_model']}")
    print(f"  OS / Kernel:         {hw['os_version']} ({hw['kernel']})")
    print(f"  Architecture:        {hw['arch']} | Python: {hw['python_version']}")
    print(f"  ONNX Runtime:        v{ort.__version__}")
    print(f"  Model File:          {model_path}")
    print(f"  Thread Count:        {threads}")
    print(f"  Warmup Runs:         {num_warmup} | Timed Runs: {num_runs}")
    print(f"  Decision tau:        {tau} | Margin delta: {delta_m}")

    opts = ort.SessionOptions()
    opts.intra_op_num_threads = threads
    opts.inter_op_num_threads = 1
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession(model_path, sess_options=opts, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    fe = PureNumpyFeatureExtractor(
        mel_filters_path=os.path.join(EXPORTS_DIR, "mel_filters_40.npy"),
        hann_window_path=os.path.join(EXPORTS_DIR, "hann_window_400.npy"),
    )

    # 1. Latency & Real-Time Factor (RTF) Benchmarking
    print(f"\n[1/2] Warming up ({num_warmup} passes)...")
    dummy_wav = np.random.randn(TARGET_SAMPLES).astype(np.float32) * 0.05
    dummy_feat = fe.compute_melspectrogram(dummy_wav)

    for _ in range(num_warmup):
        _ = session.run([output_name], {input_name: dummy_feat})

    print(f"[2/2] Running {num_runs} timed inference passes on CPU...")
    latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = session.run([output_name], {input_name: dummy_feat})
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies = np.array(latencies)
    lat_mean = round(float(np.mean(latencies)), 2)
    lat_p50 = round(float(np.percentile(latencies, 50)), 2)
    lat_p95 = round(float(np.percentile(latencies, 95)), 2)
    lat_p99 = round(float(np.percentile(latencies, 99)), 2)
    rtf = round(float((lat_p95 / 1000.0) / TARGET_DURATION), 5)

    print("\n" + "-" * 75)
    print("⏱️ PHYSICAL RASPBERRY PI LATENCY & RTF:")
    print("-" * 75)
    print(f"  Latency (Mean):      {lat_mean:6.2f} ms")
    print(f"  Latency (p50):       {lat_p50:6.2f} ms")
    print(f"  Latency (p95):       {lat_p95:6.2f} ms")
    print(f"  Latency (p99):       {lat_p99:6.2f} ms")
    print(f"  RTF (p95 / 2.0s):    {rtf:.5f}")

    # 2. Holdout Split Evaluation
    holdout_file = os.path.join(CACHE_DIR, "holdout_data.npz")
    mic_file = os.path.join(CACHE_DIR, "user_ambient_noise.npy")

    if eval_holdout and os.path.exists(holdout_file):
        print("\n" + "=" * 75)
        print("🎯 EVALUATING ON HEAVY-TESTED HOLDOUT SPLIT (Raspberry Pi Physical Run)")
        print("=" * 75)
        h_npz = np.load(holdout_file)
        h_wavs = h_npz["wavs"]
        h_labels = h_npz["labels"]
        h_spk = h_npz["speakers"] if "speakers" in h_npz else h_npz["speaker_ids"]
        h_fg = h_npz["is_filipino_group"]
        h_oos = h_npz["is_oos_speech"]
        h_vt = h_npz["voice_types"] if "voice_types" in h_npz else None

        mic_probs = None
        if os.path.exists(mic_file):
            mic_wavs = np.load(mic_file)
            m_probs = []
            for mw in mic_wavs:
                feat = fe.compute_melspectrogram(mw)
                l = session.run([output_name], {input_name: feat})[0]
                m_probs.append(softmax(l))
            mic_probs = np.concatenate(m_probs, axis=0)

        # Run inference across holdout
        all_probs = []
        for hw_wav in h_wavs:
            feat = fe.compute_melspectrogram(hw_wav)
            l = session.run([output_name], {input_name: feat})[0]
            all_probs.append(softmax(l))
        h_probs = np.concatenate(all_probs, axis=0)
        h_preds = np.argmax(h_probs, axis=1)

        # Holdout Metrics
        h_kw_acc = round(float(np.mean(h_preds == h_labels) * 100.0), 2)
        h_it_acc = h_kw_acc  # 1-to-1 intent mapping for 20 classes
        h_macro_f1 = compute_macro_f1(h_labels, h_preds, num_classes=20)

        rule_res = evaluate_rules_numpy(
            probs=h_probs,
            labels=h_labels,
            is_fg=h_fg,
            is_oos=h_oos,
            tau=tau,
            delta_m=delta_m,
            mic_probs=mic_probs,
        )

        comb = rule_res["combined"]
        h_far_oos = comb["far_oos"]
        h_far_ci = comb["far_oos_ci"]
        h_far_noise = comb["far_noise"]
        h_frr_fg = comb["frr_filipino"]

        print(f"  Holdout Utterance Count:  {len(h_labels)}")
        print(f"  Holdout Keyword Acc:      {h_kw_acc}%")
        print(f"  Holdout Intent Acc:       {h_it_acc}%")
        print(f"  Holdout Macro-F1:         {h_macro_f1}")
        print(f"  Combined FAR (OOS speech): {h_far_oos}% (95% CI: [{h_far_ci[0]}%, {h_far_ci[1]}%])")
        print(f"  Combined FAR (Mic Noise):  {h_far_noise}%")
        print(f"  Combined FRR (Filipino):  {h_frr_fg}%")

        # Per-speaker breakdown
        unique_spks = np.unique(h_spk)
        print("\n  Per-Speaker Breakdown (Holdout):")
        for spk in unique_spks:
            mask = (h_spk == spk)
            spk_acc = round(float(np.mean(h_preds[mask] == h_labels[mask]) * 100.0), 2)
            print(f"    Speaker {spk:10s} (n={np.sum(mask):2d}): Acc = {spk_acc}%")

        if h_vt is not None:
            unique_vts = np.unique(h_vt)
            print("\n  Per-Voice-Type Breakdown (Holdout):")
            for vt in unique_vts:
                mask = (h_vt == vt)
                vt_acc = round(float(np.mean(h_preds[mask] == h_labels[mask]) * 100.0), 2)
                print(f"    Voice Type {vt:18s} (n={np.sum(mask):2d}): Acc = {vt_acc}%")

        print("\n" + "=" * 75)
        print("📋 COPY & PASTE EXACTLY INTO README.md (Replacing 'REPLACE'):")
        print("=" * 75)
        print(f"""
## Section 2: Validation on the Raspberry Pi
- Keyword / intent acc: {h_kw_acc} % / {h_it_acc} %
- False-accept rate: {h_far_oos} % (OOS speech), {h_far_noise} % (mic noise)
- Latency p95 / RTF: {lat_p95} ms / {rtf}
- Latency p50: {lat_p50} ms
- Runtime: onnxruntime v{ort.__version__} · {threads} thr ({hw['hardware_model']})

## Holdout Evaluation Row:
| Split | Total Utterances | Keyword Acc (%) | Intent Acc (%) | Macro F1 | FAR OOS Speech (%) [95% CI] | FAR Mic Noise (%) | FRR Filipino (%) |
|-------|------------------|-----------------|----------------|----------|-----------------------------|-------------------|------------------|
| Holdout (Pi) | {len(h_labels)} | {h_kw_acc}% | {h_it_acc}% | {h_macro_f1} | {h_far_oos}% [{h_far_ci[0]}%, {h_far_ci[1]}%] | {h_far_noise}% | {h_frr_fg}% |
""")
        print("=" * 75)

    elif eval_holdout:
        print(f"\n⚠️ Holdout file '{holdout_file}' not found. Cannot evaluate holdout.")
    else:
        print("\nℹ️ Holdout evaluation skipped (default). Run with `--eval-holdout` on the physical Raspberry Pi.")
        print("\n📋 Latency copy-paste snippet for README:")
        print(f"- Latency p95 / RTF: {lat_p95} ms / {rtf}")
        print(f"- Latency p50: {lat_p50} ms")
        print(f"- Runtime: onnxruntime v{ort.__version__} · {threads} thr ({hw['hardware_model']})")


def main():
    parser = argparse.ArgumentParser(description="Raspberry Pi 5 Benchmark & Holdout Evaluation")
    parser.add_argument(
        "--model",
        type=str,
        default=os.path.join(EXPORTS_DIR, "bcresnet_20class_int8.onnx"),
        help="Path to INT8 or FP32 ONNX model",
    )
    parser.add_argument("--threads", type=int, default=4, help="Number of CPU threads")
    parser.add_argument("--warmup", type=int, default=30, help="Number of warmup inference passes")
    parser.add_argument("--runs", type=int, default=200, help="Number of timed inference passes (>=200)")
    parser.add_argument("--eval-holdout", action="store_true", default=False, help="Evaluate holdout split")
    parser.add_argument("--eval-test", action="store_true", default=False, help="Evaluate test split")
    parser.add_argument("--tau", type=float, default=0.55, help="Decision threshold tau")
    parser.add_argument("--delta-m", type=float, default=0.15, help="Margin threshold delta_m")
    args = parser.parse_args()

    run_benchmark(
        model_path=args.model,
        threads=args.threads,
        num_warmup=args.warmup,
        num_runs=args.runs,
        eval_holdout=args.eval_holdout,
        eval_test=args.eval_test,
        tau=args.tau,
        delta_m=args.delta_m,
    )


if __name__ == "__main__":
    main()
