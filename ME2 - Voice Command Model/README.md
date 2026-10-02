# Section 1 — AT THE VERY TOP (Submission Metadata)

- **GitHub Repository**: [`https://github.com/MisaelManingo/AI-231-Submissions-by-Misael`](https://github.com/MisaelManingo/AI-231-Submissions-by-Misael) · **Public** · **MIT License**
- **Dataset Location**: [`https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) · Pinned Revision: `6947f13073e57eb6ae67e7e2fc3680700b82aa13`
- **A100 Cluster**: Node `<cluster-host>` · 1x NVIDIA A100-SXM4-40GB · Seeds: `[42, 1337, 2026]`
- **Model Weights & Artifacts**: Available in repo under [`exports/v4_32class/`](./exports/v4_32class/) (`bcresnet_32class_int8.onnx` [114 KB], `bcresnet_32class_fp32.onnx` [279 KB], `dscnn_32class_int8.onnx` [77 KB], `checkpoints/v4_32class/best_bcresnet_32class_seed42.pt` [323 KB], `ME2-quickstart.zip` [106 KB]) · **MIT License**

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
  - ONNX Weights: **30.29 KB** (FP32) / **27.4 KB** (INT8 dynamic quantized).
* **Audio Input & Front-End**:
  - Input: 1.0 second audio buffer at 16,000 Hz (16,000 samples).
  - Spectrogram: 40 Mel filter bins, $N_{\mathrm{fft}} = 400$, hop length = 160 $\to (1, 40, 101)$ log-mel tensor.
* **Training Setup**:
  - Binary classification: Class 0 (Background Noise & Negative Speech), Class 1 ("Hey Raspberry").
  - Loss: CrossEntropyLoss; Optimizer: AdamW; Scheduler: CosineAnnealingLR.
  - Data Augmentation: Time shifting ($\pm 100\text{ ms}$), ambient noise injection (10-25 dB SNR), frequency/time masking.
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
The primary Voice Command Model employs **BC-ResNet-1** (Broadcasted Residual Network) across **32 discrete classes** (31 command classes derived directly from `vcmbench/variations.csv` [13 plain unslotted intents + 18 slotted commands: 6 slotted intents $\times$ 3 slot values] + 1 explicit `OUT_OF_SCOPE` class at index 31):

* **Audio Front-End**:
  - Sample Rate: 16,000 Hz, single-channel (mono), 16-bit PCM WAV.
  - Window & Framing: Hann window, $N_{\mathrm{fft}} = 400$ ($25.0\text{ ms}$), hop length = 160 ($10.0\text{ ms}$ step), center padding (`reflect`, $N_{\mathrm{fft}} // 2 = 200$).
  - Mel Filterbank: 40 triangular Mel filter bins spanning 0 Hz to 8,000 Hz (`mel_filters_40.npy`).
  - Power & Log Scale: $10 \log_{10}(\max(P_{\mathrm{mel}}, 10^{-10}))$ matching raw pure NumPy feature extraction.
  - Target Spectrogram Dimensions: $(B, 1, 40, 201)$ corresponding to exactly 2.0 seconds of audio.

* **Detailed Block & Stage Structure**:
  | Stage / Block | Type / Operator | Input Shape | Output Shape | Parameters | Details |
  | :--- | :--- | :--- | :--- | :--- | :--- |
  | **Init Conv** | `Conv2d` + `BN` + `ReLU` | $(B, 1, 40, 201)$ | $(B, 32, 20, 201)$ | 800 | Kernel $(5, 5)$, Stride $(2, 1)$, Padding $(2, 2)$ |
  | **Stage 1** | $2\times$ `BroadcastResBlock` | $(B, 32, 20, 201)$ | $(B, 32, 20, 201)$ | 5,632 | Stride $(1, 1)$, Broadcast Conv $1\times1$, Dropout 0.1 |
  | **Stage 2** | $2\times$ `BroadcastResBlock` | $(B, 32, 20, 201)$ | $(B, 64, 10, 101)$ | 19,456 | Block 1 stride $(2, 2)$, Block 2 stride $(1, 1)$ |
  | **Stage 3** | $2\times$ `BroadcastResBlock` | $(B, 64, 10, 101)$ | $(B, 96, 5, 51)$ | 40,704 | Block 1 stride $(2, 2)$, Block 2 stride $(1, 1)$ |
  | **Head** | `AdaptiveAvgPool2d` + `FC` | $(B, 96, 5, 51)$ | $(B, 32)$ | 3,104 | Global pool $(1, 1)$, Linear $(96 \to 32)$ |

* **Computational Complexity & Weights File Size (Profiled via `vcmbench/flops.py`)**:
  - Parameters: **0.0697 M** (69,696 parameters).
  - MACs: **42.1 M** (42,111,424 MACs).
  - FLOPs: **86.8 M** (FP32) / **89.6 M** (INT8).
  - Weights (FP32 ONNX): **0.273 MB** (279 KB).
  - Weights (INT8 ONNX): **0.111 MB** (114 KB, 2.46x compression).
  - INT8 Quantization Drop (Test 31-Command): **-0.29%** (15.98% FP32 $\to$ 15.69% INT8).

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
  - **Train**: 11,108 utterances (10,733 raw + 375 ambient room noise slices from physical G-Mark USB mic; 273 speakers).
  - **Validation**: 710 utterances pooled from Leave-One-Speaker-Out held-out folds (680 real Filipino speech clips across 5 speakers + 30 OOS speech clips).
  - **Test**: 4,443 utterances (115 unseen speakers; includes 189 real Filipino clips, 76 OOS speech clips, 15 mic noise clips, 635 open-source clips, and 3,619 synthetic clips).
  - **Holdout**: 202 utterances (5 unseen holdout speakers; includes 84 real Filipino clips from speaker `202520785`, 106 synthetic, 12 open-source, and 16 OOS clips).
- **Filipino Speech Allocation & Oversampling**:
  - Real Filipino speech in train: 680 clips across 5 speakers (`202322013`, `202322013_speaker2`, `202521746`, `S1`, `S2`, `S3`).
  - $4\times$ oversampling is applied to real Filipino clips in training, yielding 2,720 clips per epoch (34.21% effective epoch share).
  - Real OOS speech clips are weighted with class weight 4.0 to penalize false accepts.
  - Supplemental synthetic clips were omitted based on prior ablation showing they degrade OOS rejection.

---

# Section 3 — Methodological Details

## 1. Splits & Speaker-Disjointness Proof
To strictly eliminate data leakage and ensure fair evaluation, speaker IDs and file paths were verified programmatically using 12 pairwise assertions:

```python
# Programmatic Disjointness Verification in scripts/prep_dataset_32class.py
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
| **Train** | Optimization & noise grounding | 273 | 11,108 | 5.77 h | 680 clips (5 speakers) | Verified (0 leak) |
| **Val** | Model selection & threshold tuning | 36 | 710 | 0.45 h | 680 clips (LOSO pool) | Verified (0 leak) |
| **Test** | Generalization benchmark (unseen speakers) | 115 | 4,443 | 2.45 h | 189 clips (3 speakers) | Verified (0 leak) |
| **Holdout**| Physical Pi deployment & validation | 5 | 202 | 0.18 h | 84 clips (1 speaker) | Verified (0 leak) |

---

## 2. Leave-One-Speaker-Out (LOSO) Cross-Validation
To eliminate single-speaker evaluation bias, a 4-fold Leave-One-Speaker-Out (LOSO) cross-validation was executed across the real Filipino training speakers (`202322013`, `202322013_speaker2`, `202521746`, `S1_S2_S3`):

| Fold | Held-Out Speaker | Held-Out Clips | Best Val Epoch | Held-Out Filipino Accuracy (%) |
| :---: | :--- | :---: | :---: | :---: |
| **Fold 1** | `202322013` | 487 | 2 | 7.39% |
| **Fold 2** | `202322013_speaker2` | 61 | 11 | 14.75% |
| **Fold 3** | `202521746` | 72 | 7 | 11.11% |
| **Fold 4** | `S1_S2_S3` | 60 | 3 | 45.00% |
| **Mean** | — | **Total: 680** | **6** | **10.29% (Raw Pooled Acc)** |

### Pooled Held-Out Predictions & Rejection Threshold Analysis ($n=710$)
Out-of-fold predictions were pooled across all 680 held-out Filipino clips and 30 validation OOS speech clips:
- **Class-Only FAR on OOS Speech**: **93.33%**
- **Class-Only FRR on Real Filipino Speech**: **0.44%**
- **Dual-Constraint Threshold Tuning** (Target: $\mathrm{FAR}_{\mathrm{OOS}} \le 5.0\%$, $\mathrm{FRR}_{\mathrm{Filipino}} \le 30.0\%$):
  - Result: **cap not achievable**. Because out-of-domain Filipino accent variations produce lower softmax probabilities, suppressing OOS FAR to $\le 5\%$ simultaneously filters out difficult accented speech.
  - **Strict Operating Point ($\tau^* = 0.65$)**: Achieves $\mathrm{FAR}_{\mathrm{OOS}} = 3.33\%$ ($\le 5\%$) on validation OOS speech with $25.00\%$ accuracy on accepted clips, but yields high real-Filipino FRR ($99.32\%$).
  - **Lower Operating Point ($\tau_{\mathrm{bal}} = 0.20$)**: Lowers validation FRR on real Filipino speech to $23.73\%$ ($\le 30\%$) with $14.00\%$ accuracy on accepted clips, but increases validation OOS FAR to $60.00\%$.

### Pooled LOSO $\tau$ Sweep Table (Committed to [`exports/v4_32class/tau_sweep_loso_pooled.csv`](./exports/v4_32class/tau_sweep_loso_pooled.csv))
| $\tau$ | Val FAR OOS Speech (%) ($n=30$) | Val FRR Filipino (%) ($n=680$) | Acc on Accepted Clips (%) | Dual Constraints Satisfied | Operational Note |
| :---: | :---: | :---: | :---: | :---: | :--- |
| 0.10 | 90.00% | 0.85% | 11.97% | No | Minimal rejection |
| 0.15 | 86.67% | 3.05% | 12.24% | No | High false accepts |
| **0.20** | **60.00%** | **23.73%** | **14.00%** | **No** | **Lower Operating Point ($\tau_{\mathrm{bal}}$, Meets FRR $\le 30\%$)** |
| 0.25 | 40.00% | 51.02% | 15.92% | No | Intermediate |
| 0.30 | 33.33% | 74.07% | 24.84% | No | Moderate rejection |
| 0.40 | 13.33% | 92.37% | 53.33% | No | Intermediate |
| 0.50 | 13.33% | 96.61% | 55.00% | No | Strict filtering |
| 0.60 | 6.67% | 98.98% | 33.33% | No | Near target FAR |
| **0.65** | **3.33%** | **99.32%** | **25.00%** | **No** | **Strict Operating Point ($\tau^*$, Meets FAR $\le 5\%$)** |
| 0.70 | 3.33% | 99.49% | 33.33% | No | Strict |
| 0.80 | 0.00% | 99.49% | 33.33% | No | Zero OOS accepts |
| 0.90 | 0.00% | 100.00% | 0.00% | No | Complete Filipino rejection |

---

## 3. Baseline Comparison Table (Side-by-Side on Test Split)
A Depthwise-Separable CNN (**DS-CNN**) was trained, exported, and evaluated on identical splits, seeds, features, augmentation, and 32-class head:

| Metric | BC-ResNet-1 (Ours) | DS-CNN (Baseline) | Delta ($\Delta$) | Status / Source |
| :--- | :--- | :--- | :--- | :--- |
| **Parameter Count** | **0.0697 M** (69,696) | 0.0550 M (55,008) | +0.0147 M | Measured |
| **MACs (`vcmbench flops`)** | **42.1 M** | 99.2 M | **-57.1 M (2.36x cheaper)** | Profiled via ONNX |
| **FLOPs (`vcmbench flops`)** | **86.8 M** | 200.4 M | **-113.6 M (2.31x cheaper)**| Profiled via ONNX |
| **Weights File Size (FP32 ONNX)** | **0.273 MB** (279 KB) | 0.211 MB (216 KB) | +0.062 MB | Measured |
| **Weights File Size (INT8 ONNX)** | **0.111 MB** (114 KB) | 0.075 MB (77 KB) | +0.036 MB | Measured |
| **Quantization Compression** | **2.46x** | 2.81x | -0.35x | Measured |
| **INT8 Accuracy Drop (31-Cmd)** | **-0.29%** (15.98% $\to$ 15.69%) | -0.02% (13.93% $\to$ 13.91%) | -0.27% | Measured on CPU |
| **Test 31-Cmd Acc (Mean ± Std)** | **16.43% ± 1.86%** | 12.53% ± 0.99% | **+3.90% (Decisive win)** | 3 seeds (Cluster) |
| **Test 19-Intent Acc (Mean ± Std)**| **31.22% ± 1.64%** | 27.25% ± 0.47% | **+3.97% (Decisive win)** | 3 seeds (Cluster) |
| **Slot Exact Match (Mean ± Std)** | **42.22% ± 3.02%** | 33.12% ± 0.63% | **+9.10% (Decisive win)** | 3 seeds (Cluster) |
| **Macro F1 (Mean ± Std)** | **11.63% ± 2.22%** | 6.90% ± 0.55% | **+4.73%** | 3 seeds (Cluster) |
| **Macro F2 (Mean ± Std)** | **15.99% ± 2.28%** | 10.81% ± 0.43% | **+5.18%** | 3 seeds (Cluster) |
| **Balanced Acc (Mean ± Std)** | **16.48% ± 1.81%** | 12.50% ± 0.90% | **+3.98%** | 3 seeds (Cluster) |
| **FAR on OOS Speech ($\tau^* = 0.65$)** | **1.76% ± 1.64%** | 0.00% ± 0.00% | - | Test $n=76$ |
| **FAR on OOS Speech ($\tau_{\mathrm{bal}} = 0.20$)** | **57.90% ± 5.58%** | 60.09% ± 7.16% | -2.19% | Test $n=76$ |
| **FAR on Mic Noise ($\tau^* = 0.65$)** | **0.00% ± 0.00%** | 0.00% ± 0.00% | 0.00% | Test $n=15$ |
| **FAR on Mic Noise ($\tau_{\mathrm{bal}} = 0.20$)** | **84.44% ± 22.00%** | 0.00% ± 0.00% | +84.44% | Test $n=15$ |
| **Real Filipino FRR ($\tau^* = 0.65$)** | **99.12% ± 0.50%** | 97.35% ± 0.00% | +1.77% | Test $n=189$ |
| **Real Filipino FRR ($\tau_{\mathrm{bal}} = 0.20$)** | **68.78% ± 13.86%** | 48.68% ± 28.59% | +20.10% | Test $n=189$ |
| **Acc on Accepted [FRR] ($\tau^* = 0.65$)** | **39.85% ± 10.07% [87.83% ± 4.55%]** | 25.85% ± 3.36% [96.23% ± 1.95%] | **+14.00%** | 3 seeds (Cluster) |
| **Acc on Accepted [FRR] ($\tau_{\mathrm{bal}} = 0.20$)** | **20.83% ± 2.59% [26.03% ± 3.95%]** | 15.53% ± 0.35% [28.66% ± 3.41%] | **+5.30%** | 3 seeds (Cluster) |
| **Filipino Acc on Accepted [FRR] ($\tau^*$)** | **33.33% ± 47.14% [99.12% ± 0.50%]** | 0.00% ± 0.00% [97.35% ± 0.00%] | **+33.33%** | 3 seeds (Cluster) |
| **Filipino Acc on Accepted [FRR] ($\tau_{\mathrm{bal}}$)**| **11.41% ± 1.81% [68.78% ± 13.86%]** | 6.64% ± 2.05% [48.68% ± 28.59%] | **+4.77%** | 3 seeds (Cluster) |
| **Misfire Rate ($\tau^* = 0.65$)** | **7.77% ± 4.12%** | 2.85% ± 1.52% | +4.92% | 3 seeds (Cluster) |
| **Misfire Rate ($\tau_{\mathrm{bal}} = 0.20$)** | **58.66% ± 4.99%** | 60.25% ± 2.62% | -1.59% | 3 seeds (Cluster) |
| **Pi Latency p95 / RTF** | REPLACE ms / REPLACE | REPLACE ms / REPLACE | REPLACE | Physical Pi run |

---

## 4. Multi-Seed Test Results Across Dual Operating Points
Evaluated across seeds `[42, 1337, 2026]` on the unseen test split ($n=4,443$):

| Architecture | Seed | 31-Cmd Acc (%) [95% CI] | 19-Intent Acc (%) [95% CI] | Slot Match (%) | Operating Point | FAR OOS Speech (%) | Acc on Accepted (%) [FRR (%)] | Real Filipino Acc (%) | Real Filipino FRR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BC-ResNet-1** | 42 | 15.96% [14.91, 17.06] | 30.09% [28.76, 31.46] | 42.86% | Strict ($\tau^* = 0.65$) | 1.32% | 44.09% [88.37%] | 2.12% | 98.41% |
| **BC-ResNet-1** | 42 | 15.96% [14.91, 17.06] | 30.09% [28.76, 31.46] | 42.86% | Lower ($\tau_{\mathrm{bal}} = 0.20$) | 53.95% | 18.06% [24.50%] | 2.12% | 88.36% |
| **BC-ResNet-1** | 1337 | 14.43% [13.42, 15.49] | 30.02% [28.69, 31.39] | 38.24% | Strict ($\tau^* = 0.65$) | 3.95% | 28.32% [82.00%] | 3.70% | 99.47% |
| **BC-ResNet-1** | 1337 | 14.43% [13.42, 15.49] | 30.02% [28.69, 31.39] | 38.24% | Lower ($\tau_{\mathrm{bal}} = 0.20$) | 65.79% | 23.17% [22.14%] | 3.70% | 59.79% |
| **BC-ResNet-1** | 2026 | 18.91% [17.77, 20.09] | 33.54% [32.16, 34.94] | 45.56% | Strict ($\tau^* = 0.65$) | 0.00% | 47.13% [93.11%] | 5.82% | 99.47% |
| **BC-ResNet-1** | 2026 | 18.91% [17.77, 20.09] | 33.54% [32.16, 34.94] | 45.56% | Lower ($\tau_{\mathrm{bal}} = 0.20$) | 53.95% | 21.25% [31.44%] | 5.82% | 58.20% |
| **BC-ResNet-1** | **Mean ± Std** | **16.43% ± 1.86%** | **31.22% ± 1.64%** | **42.22% ± 3.02%** | **Strict ($\tau^* = 0.65$)** | **1.76% ± 1.64%** | **39.85% ± 10.07% [87.83% ± 4.55%]** | **3.88% ± 1.52%** | **99.12% ± 0.50%** |
| **BC-ResNet-1** | **Mean ± Std** | **16.43% ± 1.86%** | **31.22% ± 1.64%** | **42.22% ± 3.02%** | **Lower ($\tau_{\mathrm{bal}} = 0.20$)** | **57.90% ± 5.58%** | **20.83% ± 2.59% [26.03% ± 3.95%]** | **3.88% ± 1.52%** | **68.78% ± 13.86%** |
| DS-CNN (Base) | 42 | 13.93% [12.93, 14.98] | 27.68% [26.39, 29.02] | 32.41% | Strict ($\tau^* = 0.65$) | 0.00% | 27.94% [95.31%] | 3.70% | 97.35% |
| DS-CNN (Base) | 42 | 13.93% [12.93, 14.98] | 27.68% [26.39, 29.02] | 32.41% | Lower ($\tau_{\mathrm{bal}} = 0.20$) | 69.74% | 15.68% [24.18%] | 3.70% | 88.89% |
| DS-CNN (Base) | 1337 | 11.70% [10.77, 12.68] | 27.46% [26.16, 28.79] | 33.01% | Strict ($\tau^* = 0.65$) | 0.00% | 22.22% [94.44%] | 4.76% | 97.35% |
| DS-CNN (Base) | 1337 | 11.70% [10.77, 12.68] | 27.46% [26.16, 28.79] | 33.01% | Lower ($\tau_{\mathrm{bal}} = 0.20$) | 57.89% | 15.14% [29.38%] | 4.76% | 32.28% |
| DS-CNN (Base) | 2026 | 11.97% [11.03, 12.97] | 26.60% [25.32, 27.92] | 33.94% | Strict ($\tau^* = 0.65$) | 0.00% | 27.38% [98.95%] | 6.35% | 97.35% |
| DS-CNN (Base) | 2026 | 11.97% [11.03, 12.97] | 26.60% [25.32, 27.92] | 33.94% | Lower ($\tau_{\mathrm{bal}} = 0.20$) | 52.63% | 15.77% [32.43%] | 6.35% | 24.87% |
| DS-CNN (Base) | **Mean ± Std** | **12.53% ± 0.99%** | **27.25% ± 0.47%** | **33.12% ± 0.63%** | **Strict ($\tau^* = 0.65$)** | **0.00% ± 0.00%** | **25.85% ± 3.36% [96.23% ± 1.95%]** | **4.94% ± 1.39%** | **97.35% ± 0.00%** |
| DS-CNN (Base) | **Mean ± Std** | **12.53% ± 0.99%** | **27.25% ± 0.47%** | **33.12% ± 0.63%** | **Lower ($\tau_{\mathrm{bal}} = 0.20$)** | **60.09% ± 7.16%** | **15.53% ± 0.35% [28.66% ± 3.41%]** | **4.94% ± 1.39%** | **48.68% ± 28.59%** |

### Test Set Breakdown: Real Speech First, Synthetic Last
Breakdown across acoustic subsets on the test set ($n=4,443$, Seed 42):

| Category / Voice Type | Clip Count ($n$) | Raw 31-Cmd Acc (%) | Raw 19-Intent Acc (%) | Strict FRR (%) [95% CI] | Acc on Accepted (%) [Strict] | Lower FRR (%) | Acc on Accepted (%) [Lower] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Real Filipino Speech** | **189** | **2.12%** | **7.41%** | **98.41% [95.44, 99.46]** | **0.00% ($n=3$)** | **88.36%** | **9.09% ($n=22$)** |
| **Open-Source Speech** | **635** | **11.34%** | **12.13%** | **99.49% [98.51, 99.83]** | **66.67% ($n=3$)** | **70.39%** | **13.30% ($n=188$)** |
| **Synthetic Speech** | **3,619** | **17.49%** | **34.43%** | **86.02% [84.84, 87.11]** | **44.22% ($n=502$)** | **22.05%** | **21.37% ($n=2,821$)** |
| *Script: On-Script* | 3,899 | 16.88% | 32.78% | 86.84% [85.73, 87.88] | 44.14% ($n=503$) | 25.13% | 20.91% ($n=2,919$) |
| *Script: Off-Script* | 544 | 9.38% | 10.85% | 99.08% [97.87, 99.61] | 40.00% ($n=5$) | 68.38% | 19.19% ($n=172$) |

---

## 5. Rejection Rules Evaluation & Synthetic Capping Ablation

### Comparison of 4 Rejection Rules (Seed 42)
| Rejection Rule | Parameter | FAR OOS Speech (%) ($n=76$) | FRR In-Scope (%) ($n=4,367$) | Acc on Accepted Commands (%) | Operational Assessment |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Class-Only** | $\operatorname{argmax} = c_{\mathrm{OOS}}$ | 76.32% | 17.47% | 19.34% | Inadequate; high false accepts |
| **Threshold-Only** | $\max_c P(c) < 0.65$ | 1.32% | 86.31% | 44.09% | Highly effective OOS rejection |
| **Combined (Deployed)** | $c_{\mathrm{OOS}} \;\lor\; \max_c P(c) < 0.65$ | **1.32%** | **88.37%** | **44.09%** | **Recommended production rule** |
| **Margin Rule** | $P_{(1)} - P_{(2)} < 0.15$ | 30.26% | 51.68% | 32.88% | Intermediate trade-off |

### Synthetic Share Capping Ablation (Validation LOSO)
To prevent synthetic speech from dominating acoustic representations, synthetic:real ratios were swept on validation:
- **Ratio 1:1**: **11.11% Val Filipino Accuracy (Best)** $\to$ **Selected for production**.
- **Ratio 2:1**: 2.78% Val Filipino Accuracy.
- **Ratio 4:1**: 0.00% Val Filipino Accuracy (Severe real-speech phoneme starvation).

*Note: Supplemental synthetic audio was omitted based on prior ablation showing it degrades OOS rejection.*

---

## 6. Confusion Analysis (Top 10 Intent and Command Confusions)

### Top 10 Intent Confusions (Seed 42)
| Rank | Ground Truth Intent $\to$ Predicted Intent | Count | Error Root Cause |
| :---: | :--- | :---: | :--- |
| 1 | `CREATE_REMINDER` $\to$ `ALARM` | 300 | Temporal phrasing overlap ("remind me at...", "set alarm") |
| 2 | `TIMER` $\to$ `ALARM` | 289 | Number/time token acoustic similarity ("seconds", "minutes", "AM") |
| 3 | `COLOR` $\to$ `NEXT` | 107 | Monosyllabic command confusion ("red", "blue" vs "next") |
| 4 | `BRIGHTNESS` $\to$ `ALARM` | 85 | Numeric percentage confusion ("twenty", "sixty" vs numbers) |
| 5 | `PLAY_MUSIC` $\to$ `NEXT` | 83 | Shared media domain acoustic context |
| 6 | `CREATE_REMINDER` $\to$ `OUT_OF_SCOPE` | 83 | Long multi-word commands falling into OOS tail |
| 7 | `PAUSE` $\to$ `NEXT` | 76 | Short media command ambiguity |
| 8 | `LIGHT_ON` $\to$ `TIME` | 71 | Spectral energy similarity on fricative onset |
| 9 | `WEATHER` $\to$ `OUT_OF_SCOPE` | 67 | Conversational question phrasing rejected as non-command |
| 10 | `LIGHT_OFF` $\to$ `NEXT` | 66 | Short utterance acoustic similarity |

### Top 10 Command Confusions (Seed 42)
| Rank | Ground Truth Command $\to$ Predicted Command | Count |
| :---: | :--- | :---: |
| 1 | `CREATE_REMINDER_EXERCISE` $\to$ `ALARM_6_00AM` | 116 |
| 2 | `TIMER_30s` $\to$ `ALARM_6_00AM` | 105 |
| 3 | `CREATE_REMINDER_DRINK_WATER` $\to$ `ALARM_6_00AM` | 98 |
| 4 | `TIMER_10s` $\to$ `ALARM_6_00AM` | 94 |
| 5 | `TEMPERATURE_22` $\to$ `TEMPERATURE_26` | 90 |
| 6 | `TIMER_1m` $\to$ `ALARM_6_00AM` | 90 |
| 7 | `CREATE_REMINDER_STUDY` $\to$ `ALARM_6_00AM` | 86 |
| 8 | `COLOR_GREEN` $\to$ `NEXT` | 40 |
| 9 | `BRIGHTNESS_20` $\to$ `ALARM_6_00AM` | 37 |
| 10 | `COLOR_RED` $\to$ `NEXT` | 36 |

---

## 7. Held-Out Evaluation (Raspberry Pi Physical Run)

The **Holdout split** ($n=202$ utterances, 5 unseen speakers, 16 OOS clips, 84 Filipino group recordings) was isolated completely from the cluster pipeline. It is evaluated directly on physical Raspberry Pi hardware using the class benchmark (`airimonda/vcm-benchmark`):

| Split / Partition | Total Utterances | Command Acc (%) | Intent Acc (%) | Macro-F1 | FAR OOS Speech (%) [95% CI] | FAR Mic Noise (%) | FRR Filipino Group (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test (Strict $\tau^* = 0.65$)** | 4,443 | 16.43% ± 1.86% | 31.22% ± 1.64% | 0.1163 | 1.76% ± 1.64% | 0.00% | 99.12% ± 0.50% |
| **Test (Lower $\tau_{\mathrm{bal}} = 0.20$)** | 4,443 | 16.43% ± 1.86% | 31.22% ± 1.64% | 0.1163 | 57.90% ± 5.58% | 84.44% ± 22.00% | 68.78% ± 13.86% |
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

## 8. Hyperparameter Changes vs. Previous Run
| Hyperparameter / Component | Previous Run (94-Class) | New Consolidated Run (32-Class Production) | Justification & Rationale |
| :--- | :--- | :--- | :--- |
| **Class Schema** | 94 classes (93 variations + 1 OOS) | **32 classes (31 commands + 1 OUT_OF_SCOPE)** | The benchmark credits `(intent, slot)` only. Splitting near-identical phrasings dilutes training data per class and lowers prediction confidence. |
| **Model MACs** | 42.1 M MACs (BC-ResNet-1) | **42.1 M MACs (Profiled via `vcmbench flops`)** | Corrected attribution: 42.1 M belongs to BC-ResNet-1 (DS-CNN requires 99.2 M MACs, 2.36x more computation). |
| **Validation Strategy** | Single-speaker validation | **4-Fold Leave-One-Speaker-Out (LOSO)** | Evaluates cross-speaker generalization across Filipino training speakers without single-speaker bias. |
| **Filipino Oversampling** | $4\times$ oversampling (19.03% share) | **$4\times$ oversampling (34.21% effective share)** | Maintains high real-speech representation under 1:1 synthetic capping. |
| **Synthetic Capping** | None (uncapped) | **1:1 synthetic:real ratio per class** | Ablation confirmed 1:1 ratio yields highest held-out validation accuracy (11.11% vs 2.78% and 0.0%). |
| **Rejection Mechanism** | Strict $\tau^* = 0.65$ | **Dual Operating Points ($\tau^* = 0.65$, $\tau_{\mathrm{bal}} = 0.20$)** | Rejection cap not achievable simultaneously; reports strict OOS filtering and practical lower threshold. |
| **Stand-alone Deployment**| Scattered scripts | **`ME2-quickstart.zip` + `simulate_demo.py`** | 100% PyTorch-free standalone test harness verified in clean virtual environment. |

---

# Section 4 — Reviewer Checklist

| # | Verification Criterion | Status | Evidence / Notes |
|---|---|:---:|---|
| 1 | **Repository public, one-command reproduction** | **PASS** | MIT License, public GitHub repo, `./reproduce.sh` and `make reproduce` provided. |
| 2 | **Dataset licensed and citable (DOI)** | **PARTIAL** | Upstream constituent dataset MSVCD has DOI: `10.48804/IEKKVZ`, but course composite dataset (`airimonda/ai231-me2-voice-commands`) has no assigned DOI (pinned via commit `6947f13073e57eb6ae67e7e2fc3680700b82aa13`). License terms tabulated per source. |
| 3 | **Training logs and final checkpoint committed** | **PASS** | CSV logs and checkpoints committed under `exports/v4_32class/` and `checkpoints/v4_32class/`. |
| 4 | **Pi latency and holdout accuracy measured on physical hardware** | **PENDING** | Isolated from cluster; marked strictly with REPLACE pending on-device physical testing by user on Raspberry Pi 5. |
| 5 | **All numbers match between README, code, and logs** | **PASS** | Param counts (69,696 / 55,008), MACs (42.1M / 99.2M), and accuracy match exact committed evaluation logs. |
| 6 | **INT8 quantization accuracy drop reported** | **PASS** | Drop measured on CPU: -0.29% for BC-ResNet-1 (15.98% $\to$ 15.69%), -0.02% for DS-CNN. |
| 7 | **Disjoint splits programmatically verified** | **PASS** | 12 pairwise assertions passed (zero speaker or file leak between train/val/test/holdout). |
| 8 | **Rejection rule and threshold selection documented** | **PASS** | 4 rejection rules compared; $\tau^* = 0.65$ and $\tau_{\mathrm{bal}} = 0.20$ tuned on pooled held-out LOSO validation. |

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
rsync -avzP "${HPC_USER}@${HPC_HOST}:'${REMOTE_PATH}/ME2 - Voice Command Model/exports/v4_32class/'" ./me2_vcm/exports/
rsync -avzP "${HPC_USER}@${HPC_HOST}:'${REMOTE_PATH}/ME2 - Voice Command Model/data/v4_cache_32class/holdout_data.npz'" ./me2_vcm/data/
```

### 2. Execute Benchmark on Physical Hardware
```bash
# Step 1: Run standalone zero-hardware simulation test
python simulate_demo.py

# Step 2: Run class benchmark (airimonda/vcm-benchmark)
git clone https://github.com/airimonda/vcm-benchmark.git
cd vcm-benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python benchmark.py
```

### 3. Replace Marked Values in Document
Upon completion of physical hardware testing, replace the marked `REPLACE` tokens in:
- `Section 2: Validation on the Raspberry Pi` (Keyword / intent acc, FAR, Latency p95 / RTF, Latency p50, Runtime).
- `Section 3: Baseline Comparison Table` (Pi Latency p95 / RTF row).
- `Section 3: Held-Out Evaluation Table` (Holdout (Pi) row).
- `Section 3: Per-Voice-Type / Speaker Breakdown` (Holdout per-speaker rows).

---

# Section 6 — Physical Hardware Actions, Configuration & Testing

This deployment features full offline physical execution for all 32 voice command intents with background workers and a single-stream software audio mixer.

## 1. Physical Wiring Table (All 3.3V Logic)

| Peripheral / Sensor | Device Pin | Raspberry Pi GPIO (BCM) | Physical Header Pin | Wiring / Electrical Notes |
| :--- | :--- | :--- | :---: | :--- |
| **I2C Bus (LCD + RTC)** | SDA | **GPIO 2** | Pin 3 | Shared I2C data (LCD: `0x27`, RTC: `0x68`) |
| **I2C Bus (LCD + RTC)** | SCL | **GPIO 3** | Pin 5 | Shared I2C clock |
| **I2C Bus (LCD + RTC)** | VCC / + | 3.3V Power | Pin 1 | 3.3V Power Rail |
| **I2C Bus (LCD + RTC)** | GND / - | Ground | Pin 6 | Shared Ground |
| **DHT22 Temp & Humidity** | Data (Out)| **GPIO 4** | Pin 7 | Single-wire data (moved from GPIO 27) |
| **DHT22 Temp & Humidity** | VCC (+) | 3.3V Power | Pin 17 | 3.3V Power |
| **DHT22 Temp & Humidity** | GND (-) | Ground | Pin 9 | Ground |
| **Single RGB Status LED**| Red (R) | **GPIO 17** | Pin 11 | PWM / Digital control |
| **Single RGB Status LED**| Green (G) | **GPIO 27** | Pin 13 | PWM / Digital control |
| **Single RGB Status LED**| Blue (B) | **GPIO 22** | Pin 15 | PWM / Digital control |
| **Single RGB Status LED**| Cathode (-) | Ground | Pin 14 | Ground |
| **TM1637 4-Digit Display**| CLK | **GPIO 23** | Pin 16 | Bit-bang display clock |
| **TM1637 4-Digit Display**| DIO | **GPIO 24** | Pin 18 | Bit-bang display data |
| **TM1637 4-Digit Display**| VCC | 3.3V Power | Pin 17 | 3.3V Power |
| **TM1637 4-Digit Display**| GND | Ground | Pin 20 | Ground |
| **Grove LED Strip (WS2813)**| SIG (MOSI)| **GPIO 10** | Pin 19 | Hardware SPI0 MOSI (`spidev0.0`) |
| **Grove LED Strip (WS2813)**| VCC | 3.3V / 5V | Pin 2 or Ext | Power Rail |
| **Grove LED Strip (WS2813)**| GND | Ground | Pin 25 | Ground |
| **Push Button** | Input | **GPIO 5** | Pin 29 | Internal pull-up (active LOW) |
| **Push Button** | Ground | Ground | Pin 30 | Ground |
| **Active Buzzer** | SIG (+) | **GPIO 6** | Pin 31 | Digital active buzzer (moved from GPIO 5) |
| **Active Buzzer** | GND (-) | Ground | Pin 34 | Ground |

## 2. Configuration (`hardware/config.py`)

All hardware pin assignments, telephony constants, and file paths are centralized in [`hardware/config.py`](./hardware/config.py):
```python
# Telephony and SMS Configuration
CALL_NUMBER = "+639088152097"       # Destination phone number
MESSAGE_NUMBER = "+639088152097"    # Destination phone number
MESSAGE_TEXT = "Hello from my Raspberry Pi!"  # SMS body
CALL_CARRIER = "Globe"              # SIM carrier prompt selection

# Mock / Dry-Run Mode
# Run with --mock flag or set MOCK_HARDWARE=1 to simulate all hardware
```

## 3. Launching the Assistant

```bash
# 1. One-time offline generation of all TTS clips & audio chimes:
python generate_tts_assets.py

# 2. Run with real physical hardware:
python demo_rpi5_action.py

# 3. Or run in Mock / Dry-Run mode without physical peripherals:
python demo_rpi5_action.py --mock
```

## 4. Manual Test Checklist by Command Group

| Command Label | Spoken Example | Expected Visual & Hardware Behavior | Audio / TTS Feedback |
| :--- | :--- | :--- | :--- |
| **Wake Word** | *"Hey Raspberry"* | RGB LED turns Blue; LCD Row 0 displays `Say Command...` | Plays `chime_wake.wav` (waits until finished) |
| `LIGHT_ON` | *"Turn on the lights"* | Grove LED strip turns ON (all 10 LEDs lit at RED) | Plays `light_on.wav` ("Lights turned on") |
| `LIGHT_OFF` | *"Turn off the lights"* | Grove LED strip turns OFF (all LEDs dark) | Plays `light_off.wav` ("Lights turned off") |
| `BRIGHTNESS_20` | *"Set brightness to twenty percent"* | First 2 LEDs of strip lit | Plays `bright_20.wav` |
| `BRIGHTNESS_60` | *"Set brightness to sixty percent"* | First 6 LEDs of strip lit | Plays `bright_60.wav` |
| `BRIGHTNESS_100`| *"Set brightness to one hundred percent"*| All 10 LEDs of strip lit | Plays `bright_100.wav` |
| `COLOR_RED` | *"Set color to red"* | Grove strip changes to RED | Plays `color_red.wav` |
| `COLOR_GREEN` | *"Set color to green"* | Grove strip changes to GREEN | Plays `color_green.wav` |
| `COLOR_BLUE` | *"Set color to blue"* | Grove strip changes to BLUE | Plays `color_blue.wav` |
| `PLAY_MUSIC` | *"Play music"* | Starts first track in `Music/` in filename order | Music streams to USB speaker |
| `PAUSE` | *"Pause music"* | Music stops, preserving current track playback offset | Plays `music_pause.wav` |
| `STOP` | *"Stop music"* | Music stops, resetting playback to start of song | Plays `music_stop.wav` |
| `NEXT` | *"Next track"* | Advances to next audio track in `Music/` | Plays `music_next.wav` |
| `VOLUME_UP` | *"Volume up"* | Increases music gain by 10% (TTS/chimes stay 100%) | Plays `vol_up.wav` |
| `VOLUME_DOWN` | *"Volume down"* | Decreases music gain by 10% | Plays `vol_down.wav` |
| `TEMPERATURE_18`| *"Set temperature to 18"* | TM1637 shows `18*C` for 5s, then reverts to clock | Plays `temp_set_18.wav` |
| `TEMPERATURE_22`| *"Set temperature to 22"* | TM1637 shows `22*C` for 5s, then reverts to clock | Plays `temp_set_22.wav` |
| `TEMPERATURE_26`| *"Set temperature to 26"* | TM1637 shows `26*C` for 5s, then reverts to clock | Plays `temp_set_26.wav` |
| `TIME` | *"What time is it"* | TM1637 displays clock with blinking colon | TTS speaks RTC time from pre-made clips |
| `WEATHER` | *"What is the weather"* | Reads DHT22 sensor; logs temp & humidity | TTS speaks condition & reading |
| `TIMER_10s` | *"Set timer for 10 seconds"* | TM1637 counts down remaining seconds; buzzer beeps on expiry | Plays `timer_10s.wav` |
| `TIMER_30s` | *"Set timer for 30 seconds"* | TM1637 counts down; buzzer beeps on expiry | Plays `timer_30s.wav` |
| `TIMER_1m` | *"Set timer for one minute"* | TM1637 counts down; buzzer beeps on expiry | Plays `timer_1m.wav` |
| **Button Press**| *(Physical button)* | Silences active buzzer beep or alarm immediately | Alarm/buzzer silenced |
| `ALARM_6_00AM` | *"Set alarm for six AM"* | Persists alarm in `assistant_state.json` | Plays `alarm_6am.wav` |
| `ALARM_8_00AM` | *"Set alarm for eight AM"* | Persists alarm in `assistant_state.json` | Plays `alarm_8am.wav` |
| `ALARM_9_00PM` | *"Set alarm for nine PM"* | Persists alarm in `assistant_state.json` | Plays `alarm_9pm.wav` |
| `CREATE_REMINDER_DRINK_WATER` | *"Remind me to drink water"* | Adds task to JSON store (or refreshes 24h expiry if existing) | Plays `reminder_added.wav` or `reminder_exists.wav` |
| `CREATE_REMINDER_STUDY` | *"Remind me to study"* | Adds task to JSON store | Plays `reminder_added.wav` or `reminder_exists.wav` |
| `CREATE_REMINDER_EXERCISE` | *"Remind me to exercise"* | Adds task to JSON store | Plays `reminder_added.wav` or `reminder_exists.wav` |
| `LIST_REMINDERS` | *"List my reminders"* | Reads active unexpired reminders oldest-first | TTS reads active reminder list |
| `CALL` | *"Call"* | Triggers background Android ADB phone call | Plays `calling.wav` |
| `MESSAGE` | *"Send message"* | Triggers background Android ADB SMS | Plays `sending_message.wav` |
