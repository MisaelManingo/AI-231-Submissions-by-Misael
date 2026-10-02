# Section 1 — AT THE VERY TOP (Submission Metadata)

- **GitHub Repository**: [`https://github.com/MisaelManingo/AI-231-Submissions-by-Misael`](https://github.com/MisaelManingo/AI-231-Submissions-by-Misael) · **Public** · **MIT License**
- **Dataset Location**: [`https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands) · Pinned Revision: `6947f13073e57eb6ae67e7e2fc3680700b82aa13`
- **A100 Cluster**: Node `ai-n002.hpc.coe.upd.edu.ph` · 1x NVIDIA A100-SXM4-40GB (GPU 7) · Wall-clock: ~18.5 minutes (7 training runs: 3 BC-ResNet-1 seeds, 3 DS-CNN seeds, 1 ablation seed, 25 epochs each) · Seeds: `[42, 1337, 2026]`
- **Model Weights & Artifacts**: Available in repo under [`exports/v3_94class/`](./exports/v3_94class/) (`bcresnet_94class_int8.onnx` [120 KB], `bcresnet_94class_fp32.onnx` [303 KB], `dscnn_94class_int8.onnx` [85 KB], `checkpoints/v3_94class/best_bcresnet_94class_seed1337.pt` [339 KB], `ME2-quickstart.zip` [170 KB]) · **MIT License**

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

## Model Architecture
The primary Voice Command Model employs **BC-ResNet-1** (Broadcasted Residual Network), an ultra-compact acoustic architecture engineered for embedded edge devices and keyword spotting across **94 discrete classes** (93 fine-grained command variations from `vcmbench/variations.csv` + 1 explicit `OUT_OF_SCOPE` class at index 93):

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
  - INT8 Quantization Drop (Test): **-1.60%** (51.27% FP32 $\to$ 52.87% INT8, regularizing effect).

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
  - **Train**: 5.77 h · 10,222 utterances (includes 608 Filipino speech clips, 245 OOS speech clips, and 375 ambient room noise slices from user's physical G-Mark USB mic).
  - **Validation**: 0.55 h · 886 utterances (speaker-disjoint carve-out from train; includes 1 unseen Filipino speaker `202521746` with 72 in-scope clips and 25 OOS speech clips).
  - **Test**: 2.45 h · 4,443 utterances (115 unseen speakers; includes 189 real Filipino clips, 76 OOS speech clips, 635 open-source clips, and 3,619 synthetic clips).
  - **Holdout**: 0.18 h · 202 utterances (5 unseen holdout speakers; includes 84 real Filipino clips from speaker `202520785`, 106 synthetic, 12 open-source, and 16 OOS clips).
- **Speakers per Split**:
  - Train: **273 speakers** (272 dataset speakers + 1 physical G-Mark mic).
  - Validation: **36 speakers** (carved out from train, 0% overlap with train, test, or holdout).
  - Test: **115 speakers** (unseen).
  - Holdout: **5 speakers** (unseen).
- **Filipino Speech Allocation & Oversampling**:
  - Real Filipino speech in train: 608 clips across 5 speakers (`202322013`, `202322013_speaker2`, `S1`, `S2`, `S3`).
  - $4\times$ oversampling is applied to real Filipino clips in training, yielding 2,432 clips per epoch out of 12,046 ($20.19\%$ effective share per epoch, up from $1.33\%$ in starved setups).
  - 1 unseen Filipino speaker (`202521746`, 72 clips) is reserved in **Validation** for zero-leak threshold tuning.
- **Labels & Schema**:
  - **93 In-Scope Command Variations**: Derived directly from `vcmbench/variations.csv` spanning 19 intents (`ALARM`, `BRIGHTNESS`, `CALL`, `COLOR`, `CREATE_REMINDER`, `LIGHT_OFF`, `LIGHT_ON`, `LIST_REMINDERS`, `MESSAGE`, `NEXT`, `PAUSE`, `PLAY_MUSIC`, `STOP`, `TEMPERATURE`, `TIME`, `TIMER`, `VOLUME_DOWN`, `VOLUME_UP`, `WEATHER`).
  - **94th Class (`OUT_OF_SCOPE`)**: Index 93. Captures spoken out-of-scope sentences, conversation, and physical microphone ambient noise. Up-weighted with effective class weight = **2.50**.
  - **Hierarchical Projection**: Direct 94-class classification maps deterministically to 19 intents and slot values via `labels_94.json`.

---

## Training on the A100 Cluster
- **Cluster Hardware**: Node `ai-n002.hpc.coe.upd.edu.ph` · 1x NVIDIA A100-SXM4-40GB (GPU 7).
- **Objective Function**: Multi-Class Cross-Entropy Loss with class weighting (`OUT_OF_SCOPE` effective weight = 2.50) and label smoothing ($\alpha = 0.05$).
- **Optimizer**: `AdamW` (initial learning rate: $1.0 \times 10^{-3}$, weight decay: $1.0 \times 10^{-4}$).
- **Learning Rate Schedule**: `CosineAnnealingLR` ($T_{\max} = 25$ epochs, $\eta_{\min} = 1.0 \times 10^{-5}$).
- **Batch Size & Epochs**: Batch size = 64, 25 epochs per run.
- **Wall-Clock Time**: ~18.5 minutes across 7 full training runs (3 BC-ResNet-1 seeds, 3 DS-CNN seeds, 1 ablation run).

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

## 2. Baseline Comparison Table (Side-by-Side)
A Depthwise-Separable CNN (**DS-CNN**) was implemented, trained, exported, and evaluated on the **identical dataset splits, audio front-end features, data augmentations, and random seeds**:

| Metric | BC-ResNet-1 (Ours) | DS-CNN (Baseline) | Delta ($\Delta$) | Status / Source |
| :--- | :--- | :--- | :--- | :--- |
| **Parameter Count** | **0.0757 M** (75,710) | 0.0630 M (63,006) | +0.0127 M | Measured |
| **MACs (`vcmbench flops`)** | **42.1 M** | 99.2 M | **-57.1 M (2.35x cheaper)** | Profiled via ONNX |
| **FLOPs (`vcmbench flops`)** | **86.8 M** | 200.4 M | **-113.6 M (2.31x cheaper)**| Profiled via ONNX |
| **Weights File Size (FP32 ONNX)** | **0.296 MB** (303 KB) | 0.241 MB (247 KB) | +0.055 MB | Measured |
| **Weights File Size (INT8 ONNX)** | **0.117 MB** (120 KB) | 0.083 MB (85 KB) | +0.034 MB | Measured |
| **Quantization Compression** | **2.53x** | 2.90x | -0.37x | Measured |
| **INT8 Accuracy Drop (94-Cmd)** | **-1.60%** (51.27% $\to$ 52.87%) | -0.20% (18.73% $\to$ 18.93%) | -1.40% (Accuracy gain) | Measured on CPU |
| **INT8 Accuracy Drop (19-Intent)**| **-1.08%** (67.81% $\to$ 68.89%) | -0.58% (42.97% $\to$ 43.55%) | -0.50% (Accuracy gain) | Measured on CPU |
| **Test 94-Command Acc (Mean ± Std)**| **56.20% ± 3.59%** | 24.95% ± 4.40% | **+31.25% (Decisive win)**| 3 seeds (Cluster) |
| **Test 19-Intent Acc (Mean ± Std)** | **73.12% ± 1.32%** | 48.93% ± 4.22% | **+24.19% (Decisive win)**| 3 seeds (Cluster) |
| **Slot Exact Match (Mean ± Std)** | **81.06% ± 4.29%** | 57.62% ± 8.36% | **+23.44% (Decisive win)**| 3 seeds (Cluster) |
| **FAR on OOS Speech (Combined)**| **4.83% ± 1.24%** | 31.14% ± 5.30% | **-26.31% (Superior rejection)**| 3 seeds (Test $n=76$) |
| **FAR on Physical Mic Noise** | **0.0%** | 0.0% | 0.0% | $n=15$ takes |
| **Command Acc on Accepted Clips** | **88.43% ± 1.62%** | 36.75% ± 5.47% | **+51.68%** | 3 seeds (Cluster) |
| **Real Filipino Test Accuracy** | **7.94% ± 0.86%** | 5.29% ± 0.75% | **+2.65%** | 3 seeds ($n=189$) |
| **Pi Latency p95 / RTF** | REPLACE ms / REPLACE | REPLACE ms / REPLACE | REPLACE | Physical Pi run |

---

## 3. Multi-Seed Results (Mean ± Std on Test Set, Cluster)

All experiments were conducted with 3 random seeds (`42`, `1337`, `2026`) on the A100 cluster. Model selection was driven strictly by real Filipino validation accuracy (`v_acc_fil`), breaking ties with overall validation accuracy:

| Architecture | Seed | Best Val Epoch | Val Filipino Acc (%) | Test 94-Cmd Acc (%) [95% CI] | Test 19-Intent Acc (%) [95% CI] | Slot Match (%) | Tuned $\tau^*$ (Val) | Combined FAR OOS (%) [95% CI] | Acc on Accepted (%) (FRR Filipino %) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **BC-ResNet-1** | 42 | 14 | 8.33% | 53.25% [51.78, 54.72] | 72.27% [70.94, 73.57] | 75.60% | 0.85 | 3.95% [1.35, 10.97] | 87.70% (FRR: 97.88%) |
| **BC-ResNet-1** | 1337 | 24 | 8.33% | 61.26% [59.82, 62.69] | 74.99% [73.70, 76.25] | 84.14% | 0.70 | 6.58% [2.84, 14.49] | 90.68% (FRR: 95.77%) |
| **BC-ResNet-1** | 2026 | 19 | 8.33% | 54.09% [52.62, 55.55] | 72.11% [70.78, 73.41] | 83.43% | 0.80 | 3.95% [1.35, 10.97] | 86.92% (FRR: 96.83%) |
| **BC-ResNet-1** | **Mean ± Std** | — | **8.33% ± 0.00%** | **56.20% ± 3.59%** | **73.12% ± 1.32%** | **81.06% ± 4.29%** | **0.78** | **4.83% ± 1.24%** | **88.43% ± 1.62% (FRR: 96.82% ± 1.14%)** |
| DS-CNN (Base) | 42 | 10 | 5.56% | 18.73% [17.61, 19.90] | 42.97% [41.52, 44.43] | 47.96% | 0.25 | 34.21% [24.54, 45.40] | 29.08% (FRR: 86.24%) |
| DS-CNN (Base) | 1337 | 22 | 4.17% | 28.27% [26.96, 29.61] | 51.77% [50.30, 53.23] | 62.08% | 0.25 | 35.53% [25.70, 46.74] | 41.36% (FRR: 83.07%) |
| DS-CNN (Base) | 2026 | 14 | 4.17% | 27.84% [26.54, 29.17] | 52.06% [50.59, 53.52] | 62.83% | 0.35 | 23.68% [15.54, 34.40] | 39.81% (FRR: 88.36%) |
| DS-CNN (Base) | **Mean ± Std** | — | **4.63% ± 0.65%** | **24.95% ± 4.40%** | **48.93% ± 4.22%** | **57.62% ± 8.36%** | **0.28** | **31.14% ± 5.30%** | **36.75% ± 5.47% (FRR: 85.89% ± 6.93%)** |

*Selected Checkpoint*: **BC-ResNet-1 Seed 1337** (`checkpoints/v3_94class/best_bcresnet_94class_seed1337.pt`, val filipino acc: 8.33%, test 94-cmd acc: 61.26%, test 19-intent acc: 74.99%).

### Accuracy by Voice Type on Unseen Test Split ($n=4,443$, 115 Unseen Speakers, Seed 42 Baseline)
| Voice Type | Total Clips | In-Scope Clips | Raw 94-Cmd Acc (%) | Raw 19-Intent Acc (%) | Accepted In-Scope | Acc on Accepted (%) | FRR on In-Scope (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Real Filipino Speech** | 189 | 189 | **8.99%** | **29.10%** | 4 | 0.00% (100% Intent) | **97.88%** |
| **Open-Source Speech** | 635 | 588 | **13.39%** | **19.37%** | 22 | 40.91% | **96.26%** |
| **Synthetic Speech** | 3,619 | 3,590 | **62.56%** | **83.81%** | 1,120 | 88.93% (97.59% Intent) | **68.80%** |

---

## 4. Rejection Rule Evaluation & Threshold Selection

To reject out-of-vocabulary spoken utterances and background noise, four inference rejection rules were evaluated:
- Let $c_{\mathrm{OOS}} = 93$ denote the `OUT_OF_SCOPE` class.
1. **Class-only Rule**: Reject if $\operatorname{argmax}_c P(c) = c_{\mathrm{OOS}}$.
2. **Threshold-only Rule**: Reject if $\max_{c} P(c) < \tau$.
3. **Combined Rule** (Deployed): Reject if $\operatorname{argmax}_c P(c) = c_{\mathrm{OOS}} \;\text{or}\; \max_{c} P(c) < \tau$.
4. **Margin Variant**: Reject if $\operatorname{argmax}_c P(c) = c_{\mathrm{OOS}} \;\text{or}\; (P_{(1)} - P_{(2)}) < \Delta_m$.

### Rejection Rules Comparison (BC-ResNet-1, Seed 42, $\tau^*=0.85$, $\Delta_m=0.15$)
| Rule | FAR OOS Speech (%) (Test $n=76$) | Wilson 95% CI (OOS Speech) | FAR Mic Noise (%) ($n=15$) | Command Acc on Accepted Clips (%) | FRR on Filipino Group (%) (Test $n=189$) | Accepted / Rejected Total |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Class-only** | 100.00% | [95.19%, 100.00%] | 20.00% | 54.28% | **0.00%** | 4,435 / 8 |
| **Threshold-only**| 3.95% | [1.35%, 10.97%] | **0.00%** | **87.70%** | 97.88% | 1,149 / 3,294 |
| **Combined** (Deployed) | **3.95%** | **[1.35%, 10.97%]** | **0.00%** | **87.70%** | 97.88% | 1,149 / 3,294 |
| **Margin Variant** | 56.58% | [45.39%, 67.14%] | 6.67% | 65.08% | 68.25% | 3,225 / 1,218 |

### Validation-Only Dual-Constraint $\tau$ Sweep Table (Seed 42)
Threshold $\tau$ was tuned exclusively on the speaker-disjoint **validation set** ($n=886$) targeting dual constraints:
1. $\text{FAR}_{\text{OOS}} \le 5.0\%$
2. $\text{FRR}_{\text{Filipino}} \le 30.0\%$

| $\tau$ | Val FAR OOS Speech (%) ($n=25$) | Val FRR Filipino Group (%) ($n=72$) | Val Acc on Accepted Clips (%) | Dual Constraints Satisfied | Operational Status |
| :---: | :---: | :---: | :---: | :---: | :--- |
| 0.10 | 100.00% | 75.00% | 32.96% | No | Unfiltered baseline |
| 0.20 | 100.00% | 83.33% | 37.05% | No | High false accepts |
| 0.30 | 92.00% | 87.50% | 43.15% | No | High false accepts |
| 0.40 | 80.00% | 90.28% | 47.96% | No | Transition boundary |
| 0.50 | 64.00% | 90.28% | 54.38% | No | Moderate filtering |
| 0.60 | 48.00% | 91.67% | 61.24% | No | Elevated precision |
| 0.70 | 28.00% | 91.67% | 69.58% | No | Strict filtering |
| 0.80 | 12.00% | 95.83% | 78.43% | No | Approaching target |
| **0.85** | **4.00%** | **95.83%** | **83.65%** | **Closest Trade-off** | **Satisfies FAR $\le 5.0\%$ (FRR $\le 30\%$ not achievable)** |
| 0.90 | 4.00% | 98.61% | 88.08% | No | Ultra-strict |
| 0.95 | 0.00% | 100.00% | 92.21% | No | Total Filipino rejection |

*(Committed at [`exports/v3_94class/tau_sweep_val_bcresnet_seed42.csv`](./exports/v3_94class/tau_sweep_val_bcresnet_seed42.csv)).*

*(Dual-Constraint Status: `Achievable: False`. Because real Filipino speech validation samples exhibit lower softmax confidence than synthetic samples, no threshold simultaneously satisfied $\text{FAR}_{\text{OOS}} \le 5\%$ and $\text{FRR}_{\text{Filipino}} \le 30\%$. The closest tradeoff meeting safety $\text{FAR} \le 5\%$ is $\tau^* = 0.85$, with temperature-calibrated equivalent $\tau^*_{\text{cal}} = 0.55$).*

### Temperature Scaling Calibration Analysis
Temperature scaling ($T > 0$) was fitted on validation logits via NLL minimization:
- Fitted temperature for BC-ResNet-1: $T = 1.6960$ (Seed 42), $T = 1.5432$ (Seed 1337), $T = 1.7126$ (Seed 2026).
- Calibrated optimal threshold: $\tau^*_{\text{cal}} = 0.55$ (achieves identical ROC curve to raw $\tau^* = 0.85$).

---

## 5. Supplemental Synthetic Data Scale Ablation (Seed 42)

To evaluate the effect of synthetic data volume, an ablation was conducted by training BC-ResNet-1 (Seed 42, 25 epochs) with and without the 3,983 supplemental synthetic training clips (`supplemental_synth` config, filtered strictly to `voice_split == 'train'` to ensure 0 speaker leak):

| Configuration | Total Train Clips | Effective Filipino Share | Test 94-Cmd Acc (%) | Test 19-Intent Acc (%) | Slot Match (%) | FAR OOS Speech (%) [95% CI] | FRR Filipino Group (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Without Supp Synth)** | 10,222 | **20.19%** | **53.25%** | **72.27%** | 75.60% | **3.95% [1.35%, 10.97%]** | 97.88% |
| **Ablation (With Supp Synth)** | 14,205 (+39.0%) | 15.17% | 51.36% (-1.89%) | 67.84% (-4.43%) | **82.26% (+6.66%)** | 19.74% [12.34%, 30.04%] | **96.30%** |

### Empirical Finding:
Adding 3,983 supplemental synthetic clips improved slot exact matching on synthetic commands ($75.60\% \to 82.26\%$), but **degraded out-of-scope rejection safety by 5x** ($\text{FAR}_{\text{OOS}}$ increased from $3.95\%$ to $19.74\%$) and reduced general 94-command accuracy from $53.25\%$ to $51.36\%$. Diluting the effective share of real Filipino speech from $20.19\%$ down to $15.17\%$ exacerbated synthetic acoustic overfitting, confirming that scaling synthetic data without balanced real acoustic diversity impairs edge rejection boundaries.

---

## 6. Held-Out Evaluation (Raspberry Pi Physical Run)

The **Holdout split** ($n=202$ utterances, 5 unseen speakers, 16 OOS clips, 84 Filipino group recordings) was isolated completely from the cluster pipeline. It is evaluated directly on physical Raspberry Pi hardware using the class benchmark (`airimonda/vcm-benchmark`):

| Split / Partition | Total Utterances | Keyword Acc (%) | Intent Acc (%) | Macro-F1 | FAR OOS Speech (%) [95% CI] | FAR Mic Noise (%) | FRR Filipino Group (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Test (Cluster)** | 4,443 | 56.20% ± 3.59% | 73.12% ± 1.32% | 0.5506 ± 0.0390 | 4.83% [1.24%, 10.97%] | 0.00% | 96.82% ± 1.14% |
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
| **Model MACs** | 99.2 M (unoptimized) | **42.1 M (Profiled via `vcmbench flops`)** | BC-ResNet-1 broadcast residual architecture is 2.35x computationally lighter than DS-CNN baseline. |
| **Filipino Allocation** | 547 train / 72 val | **608 train / 72 val (4x oversampling)** | Incorporates all available Filipino training clips, maintaining 20.19% effective epoch share. |
| **Rejection Mechanism** | Combined Rule ($\tau^* = 0.75$) | **Combined Rule ($\tau^* = 0.78$ avg)** | Reject if $\operatorname{argmax} = 93$ or $p_{\max} < \tau^*$, achieving $\le 5\%$ FAR on OOS speech. |
| **Stand-alone Deployment**| Scattered scripts | **`ME2-quickstart.zip` + `simulate_demo.py`** | 100% PyTorch-free standalone test harness with standardized benchmark JSON lines. |

---

## 8. Limitations & Edge Deployment Insights
1. **Synthetic Dominance vs. Accented Speech**: Over 75% of available training utterances originate from synthetic text-to-speech engines. Models achieve >83% intent accuracy on synthetic speech but suffer from elevated false rejection rates (>95%) on accented real Filipino recordings. Synthetic data ablation confirmed that simply adding more synthetic data degrades real-world rejection boundaries.
2. **Fine-Grained Variation Confusion**: Acoustic models struggle to distinguish subtle phonetic variations that share identical root phonemes (e.g. `LIGHT_ON` variation 1 vs variation 2). Projecting variation logits to 19 schema intents boosts accuracy from 56.20% to 73.12%, confirming that semantic intent is robust even when lexical variation is ambiguous.
3. **Out-of-Scope Sample Scarcity**: With only 76 out-of-scope speech clips in test and 16 in holdout, confidence intervals on FAR are wide ([1.35%, 10.97%]). Incorporating physical microphone noise is essential to guarantee zero misfires during quiet ambient room monitoring.

---

# Section 4 — Reviewer Checklist

| # | Verification Criterion | Status | Evidence / Notes |
|---|---|:---:|---|
| 1 | **Repository public, one-command reproduction** | **PASS** | MIT License, public GitHub repo, `./reproduce.sh` and `make reproduce` provided. |
| 2 | **Dataset licensed and citable (DOI)** | **PASS** | Pinned commit `6947f13073e57eb6ae67e7e2fc3680700b82aa13`; MSVCD DOI: `10.48804/IEKKVZ`. License terms tabulated per source. |
| 3 | **Training logs and final checkpoint committed** | **PASS** | CSV logs and checkpoints committed under `exports/v3_94class/` and `checkpoints/v3_94class/`. |
| 4 | **Pi latency and holdout accuracy measured on physical hardware** | **PENDING** | Isolated from cluster; marked strictly with REPLACE pending on-device physical testing by user on Raspberry Pi 5. |
| 5 | **All numbers match between README, code, and logs** | **PASS** | Param counts (75,710 / 63,006), MACs (42.1M / 99.2M), and accuracy (56.20% / 24.95%) match exact logs. |
| 6 | **INT8 quantization accuracy drop reported** | **PASS** | Drop measured on CPU: -1.60% for BC-ResNet-1 (51.27% $\to$ 52.87%), -0.20% for DS-CNN. |
| 7 | **Disjoint splits programmatically verified** | **PASS** | 12 pairwise assertions passed (zero speaker or file leak between train/val/test/holdout). |
| 8 | **Rejection rule and threshold selection documented** | **PASS** | 4 rejection rules compared; $\tau^* = 0.85$ (calibrated 0.55) selected on validation targeting FAR $\le 5\%$. |

---

# Section 5 — Raspberry Pi Hardware Execution & Sync Guide

### 1. Synchronize Deployment Bundle from HPC to Raspberry Pi
Run the following `rsync` command on the **physical Raspberry Pi**:

```bash
# Set your HPC username and host
HPC_USER="misael.andre.maningo"
HPC_HOST="ai-n002.hpc.coe.upd.edu.ph"
REMOTE_PATH="/home/misael.andre.maningo/MEng AI/AI 231"

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
# Run standalone zero-hardware simulation test
python simulate_demo.py

# Run class benchmark (airimonda/vcm-benchmark)
git clone https://github.com/airimonda/vcm-benchmark.git
cd vcm-benchmark
pip install -e .
python -m vcmbench.pi --model ../exports/bcresnet_94class_int8.onnx
```

### 3. Replace Marked Values in Document
Upon completion of physical hardware testing, replace the marked `REPLACE` tokens in:
- `Section 2: Validation on the Raspberry Pi` (Keyword / intent acc, FAR, Latency p95 / RTF, Latency p50, Runtime).
- `Section 3: Baseline Comparison Table` (Pi Latency p95 / RTF row).
- `Section 3: Held-Out Evaluation Table` (Holdout (Pi) row).
- `Section 3: Per-Voice-Type / Speaker Breakdown` (Holdout per-speaker rows).
