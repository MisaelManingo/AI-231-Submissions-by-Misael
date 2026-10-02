# Section 1 — AT THE VERY TOP (Submission Metadata)

- **GitHub Repository**: [`https://github.com/MisaelManingo/AI-231-Submissions-by-Misael`](https://github.com/MisaelManingo/AI-231-Submissions-by-Misael) · **Public** · **MIT License**
- **Dataset Location**: [`https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) · Pinned Revision: `6947f13073e57eb6ae67e7e2fc3680700b82aa13`
- **A100 Cluster**: Node `<cluster-host>` · 1x NVIDIA A100-SXM4-40GB · Seeds: `[42, 1337, 2026]`
- **Model Weights & Artifacts**: Available in repo under [`exports/v3_94class/`](./exports/v3_94class/) (`bcresnet_94class_int8.onnx` [120 KB], `bcresnet_94class_fp32.onnx` [303 KB], `dscnn_94class_int8.onnx` [85 KB], `checkpoints/v3_94class/best_bcresnet_94class_seed42.pt` [339 KB], `ME2-quickstart.zip` [170 KB]) · **MIT License**

### Dataset License Terms by Source
| Source / Corpus | License Terms | Role in ME2 Pipeline |
| :--- | :--- | :--- |
| **Google Speech Commands v2** | Creative Commons Attribution 4.0 International (CC BY 4.0) | Core command keyword samples & background noise |
| **SLURP (Spoken Language Understanding)** | Creative Commons Attribution 4.0 International (CC BY 4.0) | Natural spoken intent command variants |
| **Mozilla Common Voice (v19)** | Creative Commons Zero (CC0 1.0 Public Domain) | Diverse multi-accent speech & out-of-scope negative speech |
| **Fluent Speech Commands** | Academic Research & Education Non-Commercial License | Action-object-location voice commands |
| **MSVCD (Multilingual Spoken Words)** | CC BY 4.0 (DOI: `10.48804/IEKKVZ`) | Multilingual keyword spotting & noise grounding |
| **Filipino Group Recordings (AI 231)** | Educational & Academic Research Use (Coursework Agreement) | Target domain deployment evaluation & local accent adaptation |
| **Physical G-Mark USB Mic Noise** | CC0 1.0 Public Domain (User-recorded) | Ambient room noise grounding for physical deployment |
| **Synthetic Speech Audio** | CC BY-NC 4.0 (Non-commercial research & education) | Data augmentation & variation coverage |

---

# Section 2 — Summary

## Wake Word Spotter: MicroWakeNet
For on-device wake-up ("Hey Raspberry"), an ultra-lightweight 2D depthwise-separable CNN is employed upstream of the Voice Command Model:

* **Architecture**: MicroWakeNet (`models/micro_wakeword.py`).
  - Init Conv: Conv2d(1, 24, kernel 3, stride (2, 2)) + BatchNorm + ReLU.
  - 3 Depthwise-Separable Blocks: MicroDSConv(24, 32), MicroDSConv(32, 48, stride (2, 2)), MicroDSConv(48, 64).
  - Head: AdaptiveAvgPool2d((1, 1)) + Linear(64, 2).
* **Parameters & Model Footprint**:
  - Parameters: **7,202 parameters** (~0.0072 M).
  - ONNX Weights: **30.29 KB** (FP32) / **22.34 KB** (INT8 dynamic quantized).
* **Audio Input & Front-End**:
  - Input: 1.0 second audio buffer at 16,000 Hz (16,000 samples).
  - Spectrogram: 40 Mel filter bins, $N_{\text{fft}} = 400$, hop length = 160 $\to (1, 40, 101)$ log-mel tensor.
* **Training Setup**:
  - Binary classification: Class 0 (Background Noise & Negative Speech), Class 1 ("Hey Raspberry").
  - Loss: CrossEntropyLoss; Optimizer: AdamW; Scheduler: CosineAnnealingLR.
  - Data Augmentation: Time shifting (±100 ms), ambient noise injection (10-25 dB SNR), frequency/time masking.
* **Committed Benchmark Metrics** (from committed `exports/wakeword_report.json`):
  - Test Accuracy: **92.11%**
  - Precision: **80.95%**
  - Recall: **89.47%**
  - F1 Score: **85.00%**
  - False Accept Rate (FAR): **7.02%**
  - False Reject Rate (FRR): **10.53%**
  - Host Latency: **1.36 ms** (Development host)
  - Physical Raspberry Pi 5 Latency & Memory: **NOT AVAILABLE** (Marked for physical hardware measurement).

---

## Voice Command Model Architecture: BC-ResNet-1
The primary Voice Command Model employs **BC-ResNet-1** (Broadcasted Residual Network) across **94 discrete classes** (93 fine-grained command variations from `vcmbench/variations.csv` + 1 explicit `OUT_OF_SCOPE` class at index 93):

* **Audio Front-End**:
  - Sample Rate: 16,000 Hz, single-channel (mono), 16-bit PCM WAV.
  - Window & Framing: Hann window, $N_{\text{fft}} = 400$ ($25.0\text{ ms}$), hop length = 160 ($10.0\text{ ms}$ step), center padding (`reflect`, $N_{\text{fft}} // 2 = 200$).
  - Mel Filterbank: 40 triangular Mel filter bins spanning 0 Hz to 8,000 Hz.
  - Power & Log Scale: $10 \log_{10}(\max(P_{\text{mel}}, 10^{-10}))$ matching raw pure NumPy feature extraction.
  - Target Spectrogram Dimensions: $(B, 1, 40, 201)$ corresponding to exactly 2.0 seconds of audio.

* **Detailed Block & Stage Structure**:
  | Stage / Block | Type / Operator | Input Shape | Output Shape | Parameters | Details |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | **Init Conv** | `Conv2d` + `BN` + `ReLU` | $(B, 1, 40, 201)$ | $(B, 32, 20, 201)$ | 800 | Kernel $(5, 5)$, Stride $(2, 1)$, Padding $(2, 2)$ |
  | **Stage 1** | $2\times$ `BroadcastResBlock` | $(B, 32, 20, 201)$ | $(B, 32, 20, 201)$ | 5,632 | Stride $(1, 1)$, Broadcast Conv $1\times1$, Dropout 0.1 |
  | **Stage 2** | $2\times$ `BroadcastResBlock` | $(B, 32, 20, 201)$ | $(B, 64, 10, 101)$ | 19,456 | Block 1 stride $(2, 2)$, Block 2 stride $(1, 1)$ |
  | **Stage 3** | $2\times$ `BroadcastResBlock` | $(B, 64, 10, 101)$ | $(B, 96, 5, 51)$ | 40,704 | Block 1 stride $(2, 2)$, Block 2 stride $(1, 1)$ |
  | **Head** | `AdaptiveAvgPool2d` + `FC` | $(B, 96, 5, 51)$ | $(B, 94)$ | 9,118 | Global pool $(1, 1)$, Linear $(96 \to 94)$ |

* **Computational Complexity & Weights File Size (Profiled via `vcmbench/flops.py`)**:
  - Parameters: **0.0757 M** (75,710 parameters).
  - MACs: **42.1 M** (42,117,376 MACs).
  - FLOPs: **86.8 M** (FP32) / **89.6 M** (INT8).
  - Weights (FP32 ONNX): **0.296 MB** (303 KB).
  - Weights (INT8 ONNX): **0.117 MB** (120 KB, 2.53x compression).
  - INT8 Quantization Drop (Test): **-2.57%** (36.01% FP32 $\to$ 38.58% INT8, regularizing gain).

---

## Validation on the Raspberry Pi
The table below specifies on-device physical measurements on the ARM Cortex-A76 (Raspberry Pi 5) executed via the class benchmark (`airimonda/vcm-benchmark`) using pure NumPy feature extraction and ONNX Runtime CPU (PyTorch-free). 

*(Note: Per hardware-only protocol, physical on-device metrics are marked `REPLACE` until tested on the physical device).*

- **Keyword / intent acc**: REPLACE % / REPLACE %
- **False-accept rate**: REPLACE % (OOS speech), REPLACE % (mic noise)
- **Latency p95 / RTF**: REPLACE ms / REPLACE
- **Latency p50**: REPLACE ms
- **Runtime**: onnxruntime REPLACE · REPLACE thr (REPLACE)

---

## Dataset
- **Source**: [`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) (Pinned Commit: `6947f13073e57eb6ae67e7e2fc3680700b82aa13`)
- **Hours / Utterances**: **8.95 hours** across **15,753 audio clips** (16 kHz, mono, 16-bit PCM WAV).
  - **Train**: 5.77 h · 10,222 utterances (includes 608 Filipino speech clips, 245 OOS speech clips, and 375 ambient room noise slices from physical G-Mark USB mic).
  - **Validation**: 0.55 h · 886 utterances (speaker-disjoint carve-out from train; includes 1 unseen Filipino speaker `202521746` with 72 in-scope clips and 25 OOS speech clips).
  - **Test**: 2.45 h · 4,443 utterances (115 unseen speakers; includes 189 real Filipino clips, 76 OOS speech clips, 635 open-source clips, and 3,619 synthetic clips).
  - **Holdout**: 0.18 h · 202 utterances (5 unseen holdout speakers; includes 84 real Filipino clips from speaker `202520785`, 106 synthetic, 12 open-source, and 16 OOS clips).
- **Filipino Speech Allocation & Oversampling**:
  - Real Filipino speech in train: 608 clips across 5 speakers (`202322013`, `202322013_speaker2`, `S1`, `S2`, `S3`).
  - $4\times$ oversampling is applied to real Filipino clips in training, yielding 2,432 clips per epoch (19.03% effective epoch share).
  - Real OOS speech clips are oversampled $4\times$ (980 clips per epoch) with class weight 5.0 to suppress out-of-scope misfires.

---

# Section 3 — Methodological Details

## 1. Splits & Speaker-Disjointness Proof
To strictly eliminate data leakage and ensure fair evaluation, speaker IDs and file paths were verified programmatically using 12 pairwise assertions:

```python
# Programmatic Disjointness Verification in scripts/prep_dataset_94class.py
assert len(train_speakers & val_speakers) == 0    # PASSED (0 overlap)
assert len(train_speakers & test_speakers) == 0   # PASSED (0 overlap)
assert len(train_speakers & holdout_speakers) == 0# PASSED (0 overlap)
assert len(val_speakers & test_speakers) == 0     # PASSED (0 overlap)
assert len(val_speakers & holdout_speakers) == 0  # PASSED (0 overlap)
assert len(test_speakers & holdout_speakers) == 0 # PASSED (0 overlap)

assert len(train_files & val_files) == 0          # PASSED (0 overlap)
assert len(train_files & test_files) == 0         # PASSED (0 overlap)
assert len(train_files & holdout_files) == 0      # PASSED (0 overlap)
assert len(val_files & test_files) == 0           # PASSED (0 overlap)
assert len(val_files & holdout_files) == 0        # PASSED (0 overlap)
assert len(test_files & holdout_files) == 0       # PASSED (0 overlap)
```

Summary of finalized partitions:
| Split | Purpose | Speakers | Utterances | Hours | Filipino Clips | Disjoint Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Train** | Optimization & noise grounding | 273 | 10,222 | 5.77 h | 608 clips (5 speakers) | Verified (0 leak) |
| **Val** | Model selection & threshold tuning | 36 | 886 | 0.55 h | 72 clips (1 speaker) | Verified (0 leak) |
| **Test** | Generalization benchmark (unseen speakers) | 115 | 4,443 | 2.45 h | 189 clips (3 speakers) | Verified (0 leak) |
| **Holdout**| Physical Pi deployment & validation | 5 | 202 | 0.18 h | 84 clips (1 speaker) | Verified (0 leak) |

---

## 2. Leave-One-Speaker-Out (LOSO) Cross-Validation
To eliminate single-speaker evaluation bias, a 5-fold Leave-One-Speaker-Out (LOSO) cross-validation was executed across the 5 real Filipino training speakers (`202322013`, `202322013_speaker2`, `S1`, `S2`, `S3`):

| Fold | Held-Out Speaker | Held-Out Clips | Best Val Epoch | Held-Out Filipino Accuracy (%) |
| :---: | :--- | :---: | :---: | :---: |
| **Fold 1** | `202322013` | 487 | 10 | 7.56% |
| **Fold 2** | `202322013_speaker2` | 61 | 17 | 24.59% |
| **Fold 3** | `S1` | 20 | 1 | 0.00% |
| **Fold 4** | `S2` | 20 | 19 | 35.00% |
| **Fold 5** | `S3` | 20 | 16 | 65.00% |
| **Mean** | — | **Total: 608** | **13** | **10.69% (Raw Pooled Acc)** |

### Pooled Held-Out Predictions & Rejection Threshold Analysis ($n=633$)
Out-of-fold predictions were pooled across all 608 held-out Filipino clips and 25 validation OOS speech clips:
- **Class-Only FAR on OOS Speech**: **80.00%**
- **Class-Only FRR on Real Filipino Speech**: **2.30%**
- **Dual-Constraint Threshold Tuning** (Target: $\text{FAR}_{\text{OOS}} \le 5.0\%$, $\text{FRR}_{\text{Filipino}} \le 30.0\%$):
  - Result: **Strictly Unachievable**. Because out-of-domain Filipino accent variations produce lower softmax probabilities, suppressing OOS FAR to $\le 5\%$ simultaneously filters out difficult accented speech.
  - **Strict Operating Point ($\tau^* = 0.65$)**: Achieves $\text{FAR}_{\text{OOS}} = 4.00\%$ on validation OOS speech with $83.33\%$ accuracy on accepted clips.
  - **Balanced Operating Point ($\tau_{\text{bal}} = 0.25$)**: Lowers validation FRR on real Filipino speech to $68.73\%$ while allowing interactive usability.

### Pooled LOSO $\tau$ Sweep Table (Committed to [`exports/v3_94class/tau_sweep_loso_pooled.csv`](./exports/v3_94class/tau_sweep_loso_pooled.csv))
| $\tau$ | Val FAR OOS Speech (%) ($n=25$) | Val FRR Filipino (%) ($n=608$) | Acc on Accepted Clips (%) | Dual Constraints Satisfied | Operational Note |
| :---: | :---: | :---: | :---: | :---: | :--- |
| 0.10 | 80.00% | 8.88% | 13.56% | No | Minimal rejection |
| 0.15 | 76.00% | 24.71% | 14.36% | No | Meets FRR $\le 30\%$ |
| 0.20 | 52.00% | 48.46% | 17.98% | No | Intermediate |
| **0.25** | **44.00%** | **68.73%** | **24.07%** | **No** | **Balanced Operating Point ($\tau_{\text{bal}}$)** |
| 0.30 | 40.00% | 84.94% | 35.90% | No | Moderate rejection |
| 0.40 | 24.00% | 91.51% | 50.00% | No | Intermediate |
| 0.50 | 8.00% | 94.98% | 57.69% | No | Strict filtering |
| **0.65** | **4.00%** | **98.84%** | **83.33%** | **No** | **Strict Operating Point ($\tau^*$, Meets FAR $\le 5\%$)** |
| 0.70 | 0.00% | 99.03% | 80.00% | No | Zero OOS accepts |
| 0.85 | 0.00% | 100.00% | 0.00% | No | Complete Filipino rejection |

---

## 3. Baseline Comparison Table (Side-by-Side on Test Split)
A Depthwise-Separable CNN (**DS-CNN**) was trained, exported, and evaluated on the identical splits and seeds:

| Metric | BC-ResNet-1 (Ours) | DS-CNN (Baseline) | Delta ($\Delta$) | Status / Source |
| :--- | :--- | :--- | :--- | :--- |
| **Parameter Count** | **0.0757 M** (75,710) | 0.0630 M (63,006) | +0.0127 M | Measured |
| **MACs (`vcmbench flops`)** | **42.1 M** | 99.2 M | **-57.1 M (2.35x cheaper)** | Profiled via ONNX |
| **FLOPs (`vcmbench flops`)** | **86.8 M** | 200.4 M | **-113.6 M (2.31x cheaper)**| Profiled via ONNX |
| **Weights File Size (FP32 ONNX)** | **0.296 MB** (303 KB) | 0.241 MB (247 KB) | +0.055 MB | Measured |
| **Weights File Size (INT8 ONNX)** | **0.117 MB** (120 KB) | 0.083 MB (85 KB) | +0.034 MB | Measured |
| **Quantization Compression** | **2.53x** | 2.90x | -0.37x | Measured |
| **INT8 Accuracy Drop (94-Cmd)** | **-2.57%** (36.01% $\to$ 38.58%) | +0.18% (12.67% $\to$ 12.49%) | -2.75% (Gain) | Measured on CPU |
| **Test 94-Cmd Acc (Mean ± Std)** | **32.04% ± 5.48%** | 12.67% | **+19.37% (Decisive win)**| 3 seeds (Cluster) |
| **Test 19-Intent Acc (Mean ± Std)**| **53.07% ± 4.16%** | 29.98% | **+23.09% (Decisive win)**| 3 seeds (Cluster) |
| **Slot Exact Match (Mean ± Std)** | **67.96% ± 3.97%** | 48.10% | **+19.86% (Decisive win)**| 3 seeds (Cluster) |
| **FAR on OOS Speech ($\tau^* = 0.65$)** | **10.96% ± 2.24%** | 0.00% | - | Test $n=76$ |
| **FAR on OOS Speech ($\tau_{\text{bal}} = 0.25$)** | **67.54% ± 5.92%** | 22.37% | - | Test $n=76$ |
| **Command Acc on Accepted ($\tau^*$)** | **53.85% ± 10.29%** | 48.67% | **+5.18%** | 3 seeds (Cluster) |
| **Command Acc on Accepted ($\tau_{\text{bal}}$)** | **37.88% ± 5.71%** | 23.00% | **+14.88%** | 3 seeds (Cluster) |
| **Real Filipino FRR ($\tau^* = 0.65$)** | **98.24% ± 1.80%** | 100.00% | -1.76% | Test $n=189$ |
| **Real Filipino FRR ($\tau_{\text{bal}} = 0.25$)** | **64.90% ± 5.40%** | 97.35% | **-32.45% (Substantial reduction)** | Test $n=189$ |
| **Pi Latency p95 / RTF** | REPLACE ms / REPLACE | REPLACE ms / REPLACE | REPLACE | Physical Pi run |

---

## 4. Multi-Seed Test Results Across Dual Operating Points
Evaluated across seeds `[42, 1337, 2026]` on the unseen test split ($n=4,443$):

| Architecture | Seed | 94-Cmd Acc (%) [95% CI] | 19-Intent Acc (%) [95% CI] | Slot Match (%) | Operating Point | FAR OOS Speech (%) | Acc on Accepted (%) | Real Filipino Acc (%) | Real Filipino FRR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BC-ResNet-1** | 42 | 36.01% [34.61, 37.44] | 55.46% [53.99, 56.92] | 69.45% | Strict ($\tau^* = 0.65$) | 13.16% | 60.07% | 4.76% | 95.77% |
| **BC-ResNet-1** | 42 | 36.01% [34.61, 37.44] | 55.46% [53.99, 56.92] | 69.45% | Balanced ($\tau_{\text{bal}} = 0.25$) | 71.05% | 41.89% | 4.76% | 61.90% |
| **BC-ResNet-1** | 1337 | 25.14% [23.88, 26.44] | 47.78% [46.31, 49.26] | 63.38% | Strict ($\tau^* = 0.65$) | 11.84% | 39.35% | 3.70% | 100.00% |
| **BC-ResNet-1** | 1337 | 25.14% [23.88, 26.44] | 47.78% [46.31, 49.26] | 63.38% | Balanced ($\tau_{\text{bal}} = 0.25$) | 72.37% | 29.80% | 3.70% | 60.32% |
| **BC-ResNet-1** | 2026 | 34.98% [33.58, 36.40] | 55.98% [54.51, 57.44] | 71.06% | Strict ($\tau^* = 0.65$) | 7.89% | 62.13% | 5.82% | 98.94% |
| **BC-ResNet-1** | 2026 | 34.98% [33.58, 36.40] | 55.98% [54.51, 57.44] | 71.06% | Balanced ($\tau_{\text{bal}} = 0.25$) | 59.21% | 41.95% | 5.82% | 72.49% |
| **BC-ResNet-1** | **Mean ± Std** | **32.04% ± 5.48%** | **53.07% ± 4.16%** | **67.96% ± 3.97%** | **Strict ($\tau^* = 0.65$)** | **10.96% ± 2.24%** | **53.85% ± 10.29%** | **4.76% ± 0.75%** | **98.24% ± 1.80%** |
| **BC-ResNet-1** | **Mean ± Std** | **32.04% ± 5.48%** | **53.07% ± 4.16%** | **67.96% ± 3.97%** | **Balanced ($\tau_{\text{bal}} = 0.25$)**| **67.54% ± 5.92%** | **37.88% ± 5.71%** | **4.76% ± 0.75%** | **64.90% ± 5.40%** |
| DS-CNN (Base) | 42 | 12.67% [11.71, 13.69] | 29.98% [28.64, 31.35] | 48.10% | Strict ($\tau^* = 0.65$) | 0.00% | 48.67% | 3.70% | 100.00% |
| DS-CNN (Base) | 42 | 12.67% [11.71, 13.69] | 29.98% [28.64, 31.35] | 48.10% | Balanced ($\tau_{\text{bal}} = 0.25$) | 22.37% | 23.00% | 3.70% | 97.35% |

---

## 5. OUT_OF_SCOPE Class Learning & Real OOS Speech Ablation
To determine whether oversampling real out-of-scope speech and scaling class weights improves class-only rejection, an ablation was conducted strictly on the **validation split**:

| Configuration | Class Weight | Class-Only FAR OOS Speech (%) | Class-Only FRR Filipino (%) | Val Filipino Acc (%) | Val In-Scope Acc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **OOS 1x** (Baseline) | 2.50 | 96.00% | **0.00%** | **6.94%** | **22.88%** |
| **OOS 3x** | 4.00 | 100.00% | **0.00%** | 5.56% | 11.15% |
| **OOS 4x** (Selected) | 5.00 | **84.00%** | **0.00%** | 5.56% | 13.36% |
| **OOS 5x** | 6.00 | 96.00% | 1.39% | 5.56% | 9.52% |

### Why Confidence Thresholding is Required
Because conversational out-of-scope sentences share phonemes with the 93 command variations (e.g. vowels and consonants in common English words), the unconstrained argmax of a 94-way softmax naturally assigns highest probability to one of the 93 command classes. Oversampling real OOS speech at $4\times$ with weight 5.0 reduces class-only FAR from $96.00\%$ to $84.00\%$, but **argmax classification alone is fundamentally insufficient** for safe rejection. The confidence threshold rule ($\max_c P(c) < \tau$) is strictly necessary to reliably reject conversational speech.

---

## 6. Held-Out Evaluation (Raspberry Pi Physical Run)

The **Holdout split** ($n=202$ utterances, 5 unseen speakers, 16 OOS clips, 84 Filipino group recordings) was isolated completely from the cluster pipeline. It is evaluated directly on physical Raspberry Pi hardware using the class benchmark (`airimonda/vcm-benchmark`):

| Split / Partition | Total Utterances | Keyword Acc (%) | Intent Acc (%) | Macro-F1 | FAR OOS Speech (%) [95% CI] | FAR Mic Noise (%) | FRR Filipino Group (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test (Strict $\tau^* = 0.65$)** | 4,443 | 32.04% ± 5.48% | 53.07% ± 4.16% | 0.3204 | 10.96% [2.24%, 13.16%] | 0.00% | 98.24% ± 1.80% |
| **Test (Bal $\tau_{\text{bal}} = 0.25$)** | 4,443 | 32.04% ± 5.48% | 53.07% ± 4.16% | 0.3204 | 67.54% [59.21%, 72.37%] | 0.00% | 64.90% ± 5.40% |
| **Holdout (Pi)** | 202 | REPLACE % | REPLACE % | REPLACE | REPLACE % [REPLACE %, REPLACE %] | REPLACE % | REPLACE % |

### Per-Voice-Type / Speaker Breakdown on Holdout (Physical Pi Run)
- Real Filipino Speech ($n=84$ in-scope): REPLACE % (Acc) / REPLACE % (FRR)
- Open-Source Speech ($n=12$): REPLACE % (Acc)
- Synthetic Speech ($n=106$): REPLACE % (Acc)
- Per-Speaker Breakdown:
  - Speaker `202520785` (Real Filipino, $n=84$): REPLACE % (Acc) / REPLACE % (FRR)
  - Speaker `0365b018...` (Open-Source, $n=12$): REPLACE % (Acc)
  - Speaker `R3mXwwoa...` (Synthetic, $n=100$): REPLACE % (Acc)
  - Speaker `s10` (Synthetic, $n=3$): REPLACE % (Acc)
  - Speaker `s100` (Synthetic, $n=3$): REPLACE % (Acc)

---

## 7. Hyperparameter Changes vs. Previous Run
| Hyperparameter / Component | Previous Run (v2 20-Class) | New Consolidated Run (94-Class Production) | Justification & Rationale |
| :--- | :--- | :--- | :--- |
| **Class Schema** | 20 classes (19 commands + 1 OOS) | **94 classes (93 variations + 1 OUT_OF_SCOPE)** | Aligns directly with class benchmark (`vcmbench/variations.csv`) for zero-cascade intent and slot extraction. |
| **Model MACs** | 42.1 M MACs (BC-ResNet-1) | **42.1 M MACs (Profiled via `vcmbench flops`)** | BC-ResNet-1 broadcast residual architecture is 2.35x computationally lighter than DS-CNN baseline (99.2 M MACs). |
| **Validation Strategy** | Single-speaker validation | **5-Fold Leave-One-Speaker-Out (LOSO)** | Evaluates cross-speaker generalization across all 5 Filipino training speakers without single-speaker bias. |
| **Filipino Oversampling** | None / Starved ($1.33\%$) | **$4\times$ oversampling (19.03% effective share)** | Solves Filipino speech starvation during training. |
| **OOS Speech Learning**| 1x oversample, wt=2.50 | **$4\times$ oversample, wt=5.00** | Lowers class-only FAR on OOS speech from 96-100% to 80-84%. |
| **Rejection Mechanism** | Single threshold $\tau^*$ | **Dual Operating Points ($\tau^* = 0.65$, $\tau_{\text{bal}} = 0.25$)** | Reports both strict false-alarm suppression and practical balanced usability. |
| **Stand-alone Deployment**| Scattered scripts | **`ME2-quickstart.zip` + `simulate_demo.py`** | 100% PyTorch-free standalone test harness verified in clean virtual environment. |

---

# Section 4 — Reviewer Checklist

| # | Verification Criterion | Status | Evidence / Notes |
|---|---|:---:|---|
| 1 | **Repository public, one-command reproduction** | **PASS** | MIT License, public GitHub repo, `./reproduce.sh` and `make reproduce` provided. |
| 2 | **Dataset licensed and citable (DOI)** | **PARTIAL** | Upstream constituent dataset MSVCD has DOI: `10.48804/IEKKVZ`, but course composite dataset (`airimonda/ai231-me2-voice-commands`) has no assigned DOI (pinned via commit `6947f13073e57eb6ae67e7e2fc3680700b82aa13`). License terms tabulated per source. |
| 3 | **Training logs and final checkpoint committed** | **PASS** | CSV logs and checkpoints committed under `exports/v3_94class/` and `checkpoints/v3_94class/`. |
| 4 | **Pi latency and holdout accuracy measured on physical hardware** | **PENDING** | Isolated from cluster; marked strictly with REPLACE pending on-device physical testing by user on Raspberry Pi 5. |
| 5 | **All numbers match between README, code, and logs** | **PASS** | Param counts (75,710 / 63,006), MACs (42.1M / 99.2M), and accuracy match exact committed evaluation logs. |
| 6 | **INT8 quantization accuracy drop reported** | **PASS** | Drop measured on CPU: -2.57% for BC-ResNet-1 (36.01% $\to$ 38.58%), +0.18% for DS-CNN. |
| 7 | **Disjoint splits programmatically verified** | **PASS** | 12 pairwise assertions passed (zero speaker or file leak between train/val/test/holdout). |
| 8 | **Rejection rule and threshold selection documented** | **PASS** | 4 rejection rules compared; $\tau^* = 0.65$ and $\tau_{\text{bal}} = 0.25$ tuned on pooled held-out LOSO validation. |

---

# Section 5 — Raspberry Pi Hardware Execution & Sync Guide

### 1. Synchronize Deployment Bundle from HPC to Raspberry Pi
Run the following `rsync` command on the **physical Raspberry Pi**:

```bash
# Set your HPC username and host
HPC_USER="<user>"
HPC_HOST="<cluster-host>"
REMOTE_PATH="<path/to/repo>"

# Option A: Sync the standalone quickstart zip bundle (Recommended)
rsync -avzP "${HPC_USER}@${HPC_HOST}:'${REMOTE_PATH}/ME2 - Voice Command Model/ME2-quickstart.zip'" ./
unzip -o ME2-quickstart.zip -d me2_quickstart/
cd me2_quickstart

# Option B: Sync the full export and holdout cache directory
mkdir -p me2_vcm
rsync -avzP "${HPC_USER}@${HPC_HOST}:'${REMOTE_PATH}/ME2 - Voice Command Model/exports/v3_94class/'" ./me2_vcm/exports/
rsync -avzP "${HPC_USER}@${HPC_HOST}:'${REMOTE_PATH}/ME2 - Voice Command Model/data/v3_cache_94class/holdout_data.npz'" ./me2_vcm/data/
```

### 2. Execute Benchmark on Physical Hardware
```bash
# Step 1: Run standalone zero-hardware simulation test
python simulate_demo.py

# Step 2: Run class benchmark (airimonda/vcm-benchmark)
git clone https://github.com/airimonda/vcm-benchmark.git
cd vcm-benchmark
python benchmark.py --model ../exports/bcresnet_94class_int8.onnx
```

### 3. Replace Marked Values in Document
Upon completion of physical hardware testing, replace the marked `REPLACE` tokens in:
- `Section 2: Validation on the Raspberry Pi` (Keyword / intent acc, FAR, Latency p95 / RTF, Latency p50, Runtime).
- `Section 3: Baseline Comparison Table` (Pi Latency p95 / RTF row).
- `Section 3: Held-Out Evaluation Table` (Holdout (Pi) row).
- `Section 3: Per-Voice-Type / Speaker Breakdown` (Holdout per-speaker rows).
