# Model Card: Tiny Voice Assistant Pipeline (MicroWakeNet & BC-ResNet-1 32-Class)

## Model Details
- **Model Names**: 
  - **Wake Word Detector**: MicroWakeNet (`exports/wakeword_int8.onnx`)
  - **Voice Command Model**: BC-ResNet-1 32-Class Classifier (`exports/v4_32class/bcresnet_32class_int8.onnx` / `exports/bcresnet_32class_int8.onnx`)
- **Model Version**: `v4_32class`
- **Developed by**: Misael Andre Maningo (AI 231, UP Diliman)
- **Model Types**: 
  - MicroWakeNet: Ultra-compact 2D Depthwise-Separable Convolutional Neural Network
  - BC-ResNet-1: Deep 2D CNN with Depthwise-Separable Broadcasted Residual Blocks and Sub-Spectral Temporal Normalization
- **License**: MIT License
- **Framework**: PyTorch 2.x (Training & Fine-tuning) / ONNX Runtime CPU (Pure NumPy + ONNX inference on Raspberry Pi 5)
- **Target Hardware**: Embedded Edge Linux Devices (Raspberry Pi 5 Model B, ARM Cortex-A76)

---

## Wake Word Spotter: MicroWakeNet

### Intended Use & Architecture
- **Primary Task**: Continuously stream single-channel audio to detect the activation phrase "Hey Raspberry".
- **Architecture**: MicroWakeNet (`models/micro_wakeword.py`).
  - Init Conv: Conv2d(1, 24, kernel 3, stride (2, 2), padding 1) + BatchNorm + ReLU.
  - Stage 1: MicroDSConv(24, 32, stride (1, 1)).
  - Stage 2: MicroDSConv(32, 48, stride (2, 2)).
  - Stage 3: MicroDSConv(48, 64, stride (1, 1)).
  - Head: AdaptiveAvgPool2d((1, 1)) + Linear(64, 2).
- **Parameters & Model Footprint**:
  - Parameters: **7,202 parameters** (~0.0072 M).
  - ONNX File Size: **30.29 KB** (FP32) / **27.4 KB** (INT8 dynamic quantized).
- **Input Representation**: Single-channel 16 kHz 16-bit PCM WAV. 40-bin log-Mel spectrogram ($N_{\mathrm{fft}} = 400$, hop $= 160$) yielding $(1, 40, 101)$ tensor corresponding to 1.0 second.
- **Classes**: 2 classes: Index 0 (Negative / Background Room Noise), Index 1 ("Hey Raspberry").
- **Training Setup**: CrossEntropyLoss, AdamW optimizer, cosine annealing schedule, audio augmentations (time shifts, ambient noise injection, masking).
- **Committed Metrics** (from committed `exports/wakeword_report.json`):
  - Test Accuracy: **92.11%**
  - Precision: **80.95%**
  - Recall: **89.47%**
  - F1 Score: **85.00%**
  - False Accept Rate (FAR): **7.02%**
  - False Reject Rate (FRR): **10.53%**
  - Host Latency: **1.36 ms** (Development host)
  - Physical Raspberry Pi 5 Metrics: **NOT AVAILABLE** (Marked for physical hardware measurement).

---

## Voice Command Model: BC-ResNet-1 (32 Classes)

### Intended Use & Scope
- **Primary Use**: PyTorch-free, low-latency spoken voice command classification for edge smart-home interfaces.
- **Input Representation**: Single-channel 16 kHz 16-bit PCM WAV audio. 40-bin log-Mel spectrogram ($N_{\mathrm{fft}} = 400$, hop $= 160$, per-utterance standardization). Target dimension: $(1, 40, 201)$ corresponding to 2.0 seconds.
- **Output Schema**: 32 discrete class logits:
  - 31 commands (13 plain unslotted intents + 18 slotted commands: 6 slotted intents $\times$ 3 slot values) derived directly from `vcmbench/variations.csv`.
  - 1 explicit `OUT_OF_SCOPE` class (index 31) capturing background speech, conversation, non-command audio, and G-Mark microphone ambient room noise.
- **Intent & Slot Derivation**: Each class index maps unambiguously to a unique `(intent, slot)` pair. There is no phrasing-level fragmentation or multi-stage cascade.

---

## Training Data & Disjoint Splits
- **Source**: [`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands)
- **Pinned Hugging Face Revision**: `6947f13073e57eb6ae67e7e2fc3680700b82aa13`
- **Split Breakdown**:
  - **Train**: 11,108 utterances (10,733 clean + 375 G-Mark ambient room noise slices; 273 speakers). Includes 680 real Filipino group speech clips across 5 speakers (4x oversampled, 34.21% effective epoch share), 245 out-of-scope speech clips (class weight 4.0), and 375 ambient room noise slices from physical G-Mark USB microphone.
  - **Synthetic Capping**: 1:1 synthetic:real ratio per class (selected on validation over 2:1 and 4:1 to prevent real-speech phoneme starvation).
  - **Cross-Validation**: 4-Fold Leave-One-Speaker-Out (LOSO) cross-validation across the Filipino training speakers.
  - **Test**: 4,443 utterances (115 speakers). Completely disjoint speakers and files. 189 real Filipino clips, 76 out-of-scope clips, 635 open-source clips, 3,619 synthetic clips. Tagged 3,899 on-script and 544 off-script.
  - **Holdout**: 202 utterances (5 speakers). Untouched; evaluated exclusively on physical hardware.
- **Disjointness Verification**: 12/12 speaker and file disjointness assertions passed cleanly.
- **Data Licensing**:
  - Google Speech Commands v2: CC BY 4.0
  - SLURP & MSVCD: CC BY 4.0
  - Mozilla Common Voice v19 & G-Mark Mic Noise: CC0 1.0 Public Domain
  - Fluent Speech Commands: Non-Commercial Research & Education
  - Filipino Group Recordings: AI 231 Coursework Agreement

---

## Architecture & Computational Complexity (BC-ResNet-1)
- **Architecture**: BC-ResNet-1 (base channel factor $c = 32$)
- **Parameters**: **69,696 parameters** (~0.0697 M)
- **Precision**: FP32 (`bcresnet_32class_fp32.onnx`) and INT8 dynamic quantized (`bcresnet_32class_int8.onnx`)
- **Computational Cost** (profiled via `vcmbench/flops.py` on 2.0s input):
  - Multiply-Accumulate Operations (MACs): **42.1 M** (2.36x lighter than DS-CNN baseline at 99.2 M)
  - Floating Point Operations (FLOPs): **86.8 M** (FP32) / **89.6 M** (INT8)
- **Weights Size**:
  - FP32 ONNX: **0.273 MB** (279 KB)
  - INT8 ONNX: **0.111 MB** (114 KB, 2.46x compression)

---

## Inference Rejection Pipeline & Operating Points
To prevent misfires on background speech, conversation, and acoustic transients, inference enforces a dual-stage decision rule:

$$\mathrm{Decision}(x) = \begin{cases} \mathrm{REJECT}, & \text{if } \operatorname{argmax}_c P(c \mid x) = c_{\mathrm{OOS}} \;\lor\; \max_c P(c \mid x) < \tau \\ \operatorname{argmax}_{c < 31} P(c \mid x), & \text{otherwise} \end{cases}$$

where $c_{\mathrm{OOS}} = 31$ (`OUT_OF_SCOPE`).

Threshold $\tau$ was swept on pooled held-out validation predictions (710 clips: 680 Filipino + 30 OOS speech). Under the dual constraint ($\mathrm{FAR}_{\mathrm{OOS}} \le 5\%$ and real-Filipino $\mathrm{FRR} \le 30\%$), the cap was **not achievable** simultaneously on validation. Thus, two distinct operating points are reported:
1. **Strict Operating Point ($\tau^* = 0.65$)**: Enforces $\mathrm{FAR}_{\mathrm{OOS}} \le 5\%$ ($\mathrm{Val\ FAR} = 3.33\%$). On test: 31-Command Acc = $16.43\% \pm 1.86\%$, 19-Intent Acc = $31.22\% \pm 1.64\%$, Slot Match = $42.22\% \pm 3.02\%$, OOS FAR = $1.76\% \pm 1.64\%$, Filipino FRR = $99.12\% \pm 0.50\%$, Filipino Acc on Accepted = $33.33\% \pm 47.14\%$.
2. **Lower Operating Point ($\tau_{\mathrm{bal}} = 0.20$)**: Enforces real-Filipino $\mathrm{FRR} \le 30\%$ ($\mathrm{Val\ FRR} = 23.73\%$). On test: 31-Command Acc = $16.43\% \pm 1.86\%$, 19-Intent Acc = $31.22\% \pm 1.64\%$, Slot Match = $42.22\% \pm 3.02\%$, OOS FAR = $57.90\% \pm 5.58\%$, Filipino FRR = $68.78\% \pm 13.86\%$, Filipino Acc on Accepted = $11.41\% \pm 1.81\%$.

---

## Limitations & Ethical Considerations
- **Synthetic Speech Bias**: The test set is mostly synthetic (3,619 of 4,443 clips, 81.5%), which skews synthetic-dominated aggregate metrics away from real-speech behavior.
- **Limited Phrases & Natural Off-Script Wording**: The 31 command classes reflect fixed templates. Natural off-script phrasing appears in both train and test sets, yielding lower accuracy ($9.38\%$ vs $16.88\%$ on-script).
- **Scarce Real Filipino Speech**: Real Filipino training speech is limited to 680 clips across 5 speakers (and 189 test clips across 3 speakers), making cross-speaker acoustic generalization challenging without synthetic data balancing.
- **Few Out-Of-Scope Clips**: With only 245 train and 76 test OOS clips, rejection calibration confidence intervals are wider than in-scope metrics.
- **Physical Microphone Matching**: Variations between studio condenser microphones and small USB array microphones affect high-frequency acoustic response and background SNR.
