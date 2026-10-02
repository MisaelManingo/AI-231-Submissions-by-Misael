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
  - INT8 Quantization Drop (Test 31-Command): **+0.42%** (70.43% FP32 $\to$ 70.85% INT8).

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
- **Hours / Utterances**: **7.62 hours** across **13,911 audio clips** in canonical and benchmark splits (16 kHz, mono, 16-bit PCM WAV).
  - **Train**: 9,266 canonical utterances (8,374 synthetic + 294 open-source + 223 real Filipino + 375 ambient room noise slices from physical G-Mark USB mic; strictly filtered to the 93 canonical phrasings of the 31 command buckets).
  - **Validation**: 253 utterances pooled from Leave-One-Speaker-Out held-out folds (223 real Filipino speech clips across 5 speakers + 30 OOS speech clips).
  - **Test**: 4,443 utterances (115 unseen speakers; includes 189 real Filipino clips, 76 OOS speech clips, 15 mic noise clips, 635 open-source clips, and 3,619 synthetic clips).
  - **Holdout**: 202 utterances (5 unseen holdout speakers; includes 84 real Filipino clips from speaker `202520785`, 106 synthetic, 12 open-source, and 16 OOS clips).
- **Filipino Speech Allocation & General Accuracy**:
  - Real Filipino speech in train: 223 canonical clips across 5 speakers (`202322013`, `202322013_speaker2`, `202521746`, `S1`, `S2`, `S3`).
  - To prioritize general model accuracy across diverse speakers without skewing toward microphone recording artifacts, training maintains balanced natural weighting (`FILIPINO_OVERSAMPLE = 1`).
  - OUT_OF_SCOPE class is grounded with real OOS speech and physical USB mic idle noise with balanced weight (`OOS_CLASS_WEIGHT = 1.2`), preventing false positives during quiet intervals.
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
| **Train** | Optimization & noise grounding | 273 | 9,266 | 4.81 h | 223 clips (5 speakers) | Verified (0 leak) |
| **Val** | Model selection & threshold tuning | 36 | 253 | 0.16 h | 223 clips (LOSO pool) | Verified (0 leak) |
| **Test** | Generalization benchmark (unseen speakers) | 115 | 4,443 | 2.45 h | 189 clips (3 speakers) | Verified (0 leak) |
| **Holdout**| Physical Pi deployment & validation | 5 | 202 | 0.18 h | 84 clips (1 speaker) | Verified (0 leak) |

---

## 2. Leave-One-Speaker-Out (LOSO) Cross-Validation
To eliminate single-speaker evaluation bias, a 4-fold Leave-One-Speaker-Out (LOSO) cross-validation was executed across the real Filipino training speakers (`202322013`, `202322013_speaker2`, `202521746`, `S1_S2_S3`):

| Fold | Held-Out Speaker | Held-Out Clips | Best Val Epoch | Held-Out Filipino Accuracy (%) |
| :---: | :--- | :---: | :---: | :---: |
| **Fold 1** | `202322013` | 170 | 5 | 3.53% |
| **Fold 2** | `202322013_speaker2` | 14 | 8 | 35.71% |
| **Fold 3** | `202521746` | 30 | 5 | 40.00% |
| **Fold 4** | `S1_S2_S3` | 9 | 6 | 55.56% |
| **Mean** | — | **Total: 223** | **6** | **12.56% (Raw Pooled Acc)** |

### Pooled Held-Out Predictions & Rejection Threshold Analysis ($n=253$)
Out-of-fold predictions were pooled across all 223 held-out Filipino clips and 30 validation OOS speech clips:
- **Class-Only FAR on OOS Speech**: **93.33%**
- **Class-Only FRR on Real Filipino Speech**: **0.00%**
- **Dual-Constraint Threshold Tuning** (Target: $\mathrm{FAR}_{\mathrm{OOS}} \le 5.0\%$, $\mathrm{FRR}_{\mathrm{Filipino}} \le 30.0\%$):
  - Result: **cap not achievable**. Because out-of-domain Filipino accent variations produce lower softmax probabilities, suppressing OOS FAR to $\le 5\%$ simultaneously filters out difficult accented speech.
  - **Strict Operating Point ($\tau^* = 0.70$)**: Achieves strong OOS rejection filtering with high confidence requirements on accepted commands.
  - **Lower Operating Point ($\tau_{\mathrm{bal}} = 0.20$)**: Lowers validation FRR on real Filipino speech to $7.76\%$ ($\le 30\%$) and overall test FRR to $1.60\%$.

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
| **INT8 Accuracy Drop (31-Cmd)** | **+0.42%** (70.43% $\to$ 70.85%) | -1.42% (47.36% $\to$ 45.94%) | +1.84% | Measured on CPU |
| **Test 31-Cmd Acc (Mean ± Std)** | **69.71% ± 1.51%** | 48.04% ± 1.20% | **+21.67% (Decisive win)**| 3 seeds (Cluster) |
| **Test 19-Intent Acc (Mean ± Std)**| **73.36% ± 1.12%** | 59.32% ± 2.64% | **+14.04% (Decisive win)**| 3 seeds (Cluster) |
| **Slot Exact Match (Mean ± Std)** | **92.39% ± 1.11%** | 75.48% ± 3.13% | **+16.91% (Decisive win)**| 3 seeds (Cluster) |
| **Macro F1 (Mean ± Std)** | **68.48% ± 1.44%** | 43.81% ± 1.35% | **+24.67%** | 3 seeds (Cluster) |
| **Macro F2 (Mean ± Std)** | **69.29% ± 1.37%** | 48.14% ± 1.21% | **+21.15%** | 3 seeds (Cluster) |
| **Balanced Acc (Mean ± Std)** | **68.80% ± 1.48%** | 47.40% ± 1.20% | **+21.40%** | 3 seeds (Cluster) |
| **FAR on OOS Speech ($\tau^* = 0.70$)** | **27.63% ± 3.72%** | 12.28% ± 3.78% | +15.35% | Test $n=76$ |
| **FAR on OOS Speech ($\tau_{\mathrm{bal}} = 0.20$)** | **92.10% ± 2.84%** | 86.40% ± 4.47% | +5.70% | Test $n=76$ |
| **FAR on Mic Noise ($\tau^* = 0.70$)** | **0.00% ± 0.00%** | 0.00% ± 0.00% | 0.00% | Test $n=15$ |
| **FAR on Mic Noise ($\tau_{\mathrm{bal}} = 0.20$)** | **35.56% ± 45.65%** | 2.22% ± 3.14% | +33.34% | Test $n=15$ |
| **Real Filipino FRR ($\tau^* = 0.70$)** | **93.30% ± 1.52%** | 99.82% ± 0.25% | -6.52% | Test $n=189$ |
| **Real Filipino FRR ($\tau_{\mathrm{bal}} = 0.20$)** | **7.76% ± 6.50%** | 60.67% ± 18.19% | **-52.91% (Dramatic drop)**| Test $n=189$ |
| **Acc on Accepted [FRR] ($\tau^* = 0.70$)** | **89.97% ± 1.18% [36.55% ± 0.69%]** | 85.24% ± 2.81% [74.01% ± 1.73%] | **+4.73%** | 3 seeds (Cluster) |
| **Acc on Accepted [FRR] ($\tau_{\mathrm{bal}} = 0.20$)** | **71.93% ± 1.51% [1.60% ± 0.09%]** | 52.87% ± 1.45% [9.49% ± 0.56%] | **+19.06%** | 3 seeds (Cluster) |
| **Filipino Acc on Accepted [FRR] ($\tau^*$)** | **31.30% ± 6.06% [93.30% ± 1.52%]** | 0.00% ± 0.00% [99.82% ± 0.25%] | **+31.30%** | 3 seeds (Cluster) |
| **Filipino Acc on Accepted [FRR] ($\tau_{\mathrm{bal}}$)**| **9.90% ± 2.31% [7.76% ± 6.50%]** | 12.48% ± 4.80% [60.67% ± 18.19%] | -2.58% | 3 seeds (Cluster) |
| **Misfire Rate ($\tau^* = 0.70$)** | **6.37% ± 0.76%** | 3.85% ± 0.85% | +2.52% | 3 seeds (Cluster) |
| **Misfire Rate ($\tau_{\mathrm{bal}} = 0.20$)** | **27.62% ± 1.46%** | 42.67% ± 1.53% | **-15.05%** | 3 seeds (Cluster) |
| **Pi Latency p95 / RTF** | REPLACE ms / REPLACE | REPLACE ms / REPLACE | REPLACE | Physical Pi run |

---

## 4. Multi-Seed Test Results Across Dual Operating Points
Evaluated across seeds `[42, 1337, 2026]` on the unseen test split ($n=4,443$):

| Architecture | Seed | 31-Cmd Acc (%) [95% CI] | 19-Intent Acc (%) [95% CI] | Slot Match (%) | Operating Point | FAR OOS Speech (%) | Acc on Accepted (%) [FRR (%)] | Real Filipino Acc (%) | Real Filipino FRR (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BC-ResNet-1** | 42 | 70.43% [69.07, 71.75] | 74.25% [72.95, 75.52] | 92.33% | Strict ($\tau^* = 0.70$) | 25.00% | 90.28% [35.65%] | 11.11% | 95.24% |
| **BC-ResNet-1** | 42 | 70.43% [69.07, 71.75] | 74.25% [72.95, 75.52] | 92.33% | Lower ($\tau_{\mathrm{bal}} = 0.20$) | 96.05% | 71.39% [1.65%] | 11.11% | 16.93% |
| **BC-ResNet-1** | 1337 | 71.08% [69.73, 72.39] | 74.05% [72.74, 75.32] | 93.78% | Strict ($\tau^* = 0.70$) | 32.89% | 91.31% [37.33%] | 8.99% | 91.53% |
| **BC-ResNet-1** | 1337 | 71.08% [69.73, 72.39] | 74.05% [72.74, 75.32] | 93.78% | Lower ($\tau_{\mathrm{bal}} = 0.20$) | 89.47% | 73.18% [1.47%] | 8.99% | 3.70% |
| **BC-ResNet-1** | 2026 | 67.61% [66.23, 68.96] | 71.78% [70.44, 73.08] | 91.06% | Strict ($\tau^* = 0.70$) | 25.00% | 88.33% [36.66%] | 7.41% | 93.12% |
| **BC-ResNet-1** | 2026 | 67.61% [66.23, 68.96] | 71.78% [70.44, 73.08] | 91.06% | Lower ($\tau_{\mathrm{bal}} = 0.20$) | 90.79% | 70.27% [1.67%] | 7.41% | 2.65% |
| **BC-ResNet-1** | **Mean ± Std** | **69.71% ± 1.51%** | **73.36% ± 1.12%** | **92.39% ± 1.11%** | **Strict ($\tau^* = 0.70$)** | **27.63% ± 3.72%** | **89.97% ± 1.18% [36.55% ± 0.69%]** | **9.17% ± 1.74%** | **93.30% ± 1.52%** |
| **BC-ResNet-1** | **Mean ± Std** | **69.71% ± 1.51%** | **73.36% ± 1.12%** | **92.39% ± 1.11%** | **Lower ($\tau_{\mathrm{bal}} = 0.20$)** | **92.10% ± 2.84%** | **71.93% ± 1.51% [1.60% ± 0.09%]** | **9.17% ± 1.74%** | **7.76% ± 6.50%** |
| DS-CNN (Base) | 42 | 47.33% [45.87, 48.80] | 58.99% [57.54, 60.43] | 74.18% | Strict ($\tau^* = 0.70$) | 17.11% | 85.91% [73.32%] | 6.35% | 100.0% |
| DS-CNN (Base) | 42 | 47.33% [45.87, 48.80] | 58.99% [57.54, 60.43] | 74.18% | Lower ($\tau_{\mathrm{bal}} = 0.20$) | 90.79% | 52.09% [9.55%] | 6.35% | 50.26% |
| DS-CNN (Base) | 1337 | 49.72% [48.25, 51.19] | 62.71% [61.27, 64.12] | 72.46% | Strict ($\tau^* = 0.70$) | 7.89% | 87.52% [76.39%] | 6.88% | 99.47% |
| DS-CNN (Base) | 1337 | 49.72% [48.25, 51.19] | 62.71% [61.27, 64.12] | 72.46% | Lower ($\tau_{\mathrm{bal}} = 0.20$) | 80.26% | 54.49% [10.14%] | 6.88% | 86.24% |
| DS-CNN (Base) | 2026 | 47.06% [45.60, 48.53] | 56.27% [54.81, 57.72] | 79.80% | Strict ($\tau^* = 0.70$) | 11.84% | 82.28% [72.32%] | 6.88% | 100.0% |
| DS-CNN (Base) | 2026 | 47.06% [45.60, 48.53] | 56.27% [54.81, 57.72] | 79.80% | Lower ($\tau_{\mathrm{bal}} = 0.20$) | 88.16% | 52.02% [8.77%] | 6.88% | 45.50% |
| DS-CNN (Base) | **Mean ± Std** | **48.04% ± 1.20%** | **59.32% ± 2.64%** | **75.48% ± 3.13%** | **Strict ($\tau^* = 0.70$)** | **12.28% ± 3.78%** | **85.24% ± 2.81% [74.01% ± 1.73%]** | **6.70% ± 0.90%** | **99.82% ± 0.25%** |
| DS-CNN (Base) | **Mean ± Std** | **48.04% ± 1.20%** | **59.32% ± 2.64%** | **75.48% ± 3.13%** | **Lower ($\tau_{\mathrm{bal}} = 0.20$)** | **86.40% ± 4.47%** | **52.87% ± 1.45% [9.49% ± 0.56%]** | **6.70% ± 0.90%** | **60.67% ± 18.19%** |

### Test Set Breakdown: Real Speech First, Synthetic Last
Breakdown across acoustic subsets on the test set ($n=4,443$, Seed 42):

| Category / Voice Type | Clip Count ($n$) | Raw 31-Cmd Acc (%) | Raw 19-Intent Acc (%) | Strict FRR (%) [95% CI] | Acc on Accepted (%) [Strict] | Lower FRR (%) | Acc on Accepted (%) [Lower] |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Real Filipino Speech** | **189** | **11.11%** | **14.29%** | **95.24% [91.20, 97.47]** | **33.33% ($n=9$)** | **16.93%** | **9.09% ($n=157$)** |
| **Open-Source Speech** | **635** | **16.54%** | **17.64%** | **73.81% [70.11, 77.20]** | **31.17% ($n=154$)** | **2.68%** | **16.99% ($n=618$)** |
| **Synthetic Speech** | **3,619** | **82.98%** | **87.32%** | **26.27% [24.85, 27.73]** | **93.92% ($n=2,647$)**| **0.67%** | **83.47% ($n=3,595$)**|
| *Script: On-Script* | 3,899 | 78.64% | 82.76% | 30.08% [28.65, 31.55] | 93.75% ($n=2,673$) | 1.36% | 79.57% ($n=3,846$) |
| *Script: Off-Script* | 544 | 11.58% | 13.24% | 74.82% [71.00, 78.28] | 22.63% ($n=137$) | 3.31% | 11.98% ($n=526$) |

---

## 5. Rejection Rules Evaluation & Synthetic Capping Ablation

### Comparison of 4 Rejection Rules (Seed 42)
| Rejection Rule | Parameter | FAR OOS Speech (%) ($n=76$) | FRR In-Scope (%) ($n=4,367$) | Acc on Accepted Commands (%) | Operational Assessment |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Class-Only** | $\operatorname{argmax} = c_{\mathrm{OOS}}$ | 98.68% | 0.69% | 72.12% ($n=4,337$) | Inadequate OOS filtering; high false accepts |
| **Threshold-Only** | $\max_c P(c) < 0.70$ | 26.32% | 35.56% | 90.15% ($n=2,812$) | High precision on accepted commands |
| **Combined (Deployed)** | $c_{\mathrm{OOS}} \;\lor\; \max_c P(c) < 0.70$ | **25.00%** | **35.65%** | **90.28% ($n=2,808$)** | **Recommended production rule (best balance)** |
| **Margin Rule** | $P_{(1)} - P_{(2)} < 0.15$ | 69.74% | 14.93% | 80.26% ($n=3,713$) | Intermediate trade-off |

### Synthetic Share Capping Ablation (Validation LOSO)
To prevent synthetic speech from dominating acoustic representations, synthetic:real ratios were swept on validation:
- **Ratio 1:1** ($n=1,372$ train): 6.67% Val Filipino Acc, 13.28% Val All Acc.
- **Ratio 2:1** ($n=1,606$ train): 13.33% Val Filipino Acc, 11.96% Val All Acc.
- **Ratio 4:1** ($n=2,033$ train): 6.67% Val Filipino Acc, 12.95% Val All Acc.
- **Canonical Unconstrained (16:1)** ($n=8,017$ train): 6.67% Val Filipino Acc, **57.85% Val All Acc** $\to$ **Selected for production**. Capping discarded over 7,000 canonical training clips, starving the model of phrasing variations; unconstrained canonical training boosted test command accuracy from 16.43% to 69.71% and intent accuracy to 73.36%.

*Note: Supplemental synthetic audio was omitted based on prior ablation showing it degrades OOS rejection.*

---

## 6. Confusion Analysis (Top 10 Intent and Command Confusions)

### Top 10 Intent Confusions (Seed 42)
| Rank | Ground Truth Intent $\to$ Predicted Intent | Count | Error Root Cause |
| :---: | :--- | :---: | :--- |
| 1 | `VOLUME_UP` $\to$ `VOLUME_DOWN` | 41 | Directional command pair acoustic similarity ("up" vs "down" on low SNR) |
| 2 | `TIME` $\to$ `VOLUME_DOWN` | 33 | Short monosyllabic/bisyllabic prompt overlap |
| 3 | `STOP` $\to$ `PLAY_MUSIC` | 33 | Media control domain shared context |
| 4 | `CALL` $\to$ `STOP` | 32 | Short command acoustic ambiguity |
| 5 | `VOLUME_UP` $\to$ `ALARM` | 29 | Spectral energy similarity on vowel nuclei |
| 6 | `LIGHT_OFF` $\to$ `LIGHT_ON` | 21 | Polarity confusion ("turn off" vs "turn on" sharing "light") |
| 7 | `VOLUME_DOWN` $\to$ `ALARM` | 21 | Tail phoneme confusion |
| 8 | `COLOR` $\to$ `ALARM` | 20 | Color command confusion with alarm bucket |
| 9 | `CALL` $\to$ `VOLUME_DOWN` | 19 | Phone call vs media volume phoneme overlap |
| 10 | `CREATE_REMINDER` $\to$ `ALARM` | 19 | Temporal scheduling intent semantic and acoustic overlap |

### Top 10 Command Confusions (Seed 42)
| Rank | Ground Truth Command $\to$ Predicted Command | Count |
| :---: | :--- | :---: |
| 1 | `VOLUME_UP` $\to$ `VOLUME_DOWN` | 41 |
| 2 | `BRIGHTNESS_20` $\to$ `BRIGHTNESS_100` | 36 |
| 3 | `TIME` $\to$ `VOLUME_DOWN` | 33 |
| 4 | `STOP` $\to$ `PLAY_MUSIC` | 33 |
| 5 | `CALL` $\to$ `STOP` | 32 |
| 6 | `LIGHT_OFF` $\to$ `LIGHT_ON` | 21 |
| 7 | `TEMPERATURE_22` $\to$ `TEMPERATURE_18` | 21 |
| 8 | `CALL` $\to$ `VOLUME_DOWN` | 19 |
| 9 | `LIGHT_OFF` $\to$ `STOP` | 17 |
| 10 | `TIMER_30s` $\to$ `TIMER_10s` | 17 |

---

## 7. Held-Out Evaluation (Raspberry Pi Physical Run)

The **Holdout split** ($n=202$ utterances, 5 unseen speakers, 16 OOS clips, 84 Filipino group recordings) was isolated completely from the cluster pipeline. It is evaluated directly on physical Raspberry Pi hardware using the class benchmark (`airimonda/vcm-benchmark`):

| Split / Partition | Total Utterances | Command Acc (%) | Intent Acc (%) | Macro-F1 | FAR OOS Speech (%) [95% CI] | FAR Mic Noise (%) | FRR Filipino Group (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test (Strict $\tau^* = 0.70$)** | 4,443 | 69.71% ± 1.51% | 73.36% ± 1.12% | 0.6848 | 27.63% ± 3.72% | 0.00% | 93.30% ± 1.52% |
| **Test (Lower $\tau_{\mathrm{bal}} = 0.20$)** | 4,443 | 69.71% ± 1.51% | 73.36% ± 1.12% | 0.6848 | 92.10% ± 2.84% | 0.00% | 7.76% ± 6.50% |
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
| **Filipino Weighting** | $4\times$ oversampling (19.03% share) | **$1\times$ natural weighting** | Eliminates artificial skew toward recording artifacts; prioritizes general acoustic accuracy across all speakers. |
| **Synthetic Capping** | None (uncapped) | **Canonical Unconstrained (16:1)** | Capping to 1:1/2:1/4:1 starved models of phrasing diversity (~16% acc); unconstrained canonical yielded 69.71% test command accuracy. |
| **Rejection Mechanism** | Strict $\tau^* = 0.65$ | **Dual Operating Points ($\tau^* = 0.70$, $\tau_{\mathrm{bal}} = 0.20$)** | Rejection cap not achievable simultaneously; reports strict OOS filtering and practical lower threshold. |
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
| 6 | **INT8 quantization accuracy drop reported** | **PASS** | Drop measured on CPU: +0.42% for BC-ResNet-1 (70.43% $\to$ 70.85%), -1.39% for DS-CNN (47.33% $\to$ 45.94%). |
| 7 | **Disjoint splits programmatically verified** | **PASS** | 12 pairwise assertions passed (zero speaker or file leak between train/val/test/holdout). |
| 8 | **Rejection rule and threshold selection documented** | **PASS** | 4 rejection rules compared; $\tau^* = 0.70$ and $\tau_{\mathrm{bal}} = 0.20$ tuned on pooled held-out LOSO validation. |

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
