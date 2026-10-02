# Model Card: Tiny Voice Assistant Pipeline (MicroWakeNet & BC-ResNet-1 94-Class)

## Model Details
- **Model Names**: 
  - **Wake Word Detector**: MicroWakeNet (`exports/wakeword_int8.onnx`)
  - **Voice Command Model**: BC-ResNet-1 94-Class Classifier (`exports/v3_94class/bcresnet_94class_int8.onnx`)
- **Model Version**: `v3_94class`
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
  - ONNX File Size: **30.29 KB** (FP32) / **22.34 KB** (INT8 dynamic quantized).
- **Input Representation**: Single-channel 16 kHz 16-bit PCM WAV. 40-bin log-Mel spectrogram ($N_{\text{fft}} = 400$, hop $= 160$) yielding $(1, 40, 101)$ tensor corresponding to 1.0 second.
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

## Voice Command Model: BC-ResNet-1 (94 Classes)

### Intended Use & Scope
- **Primary Use**: PyTorch-free, low-latency spoken voice command classification for edge smart-home interfaces.
- **Input Representation**: Single-channel 16 kHz 16-bit PCM WAV audio. 40-bin log-Mel spectrogram ($N_{\text{fft}} = 400$, hop $= 160$, per-utterance standardization). Target dimension: $(1, 40, 201)$ corresponding to 2.0 seconds.
- **Output Schema**: 94 discrete class logits:
  - 93 specific command variations spanning 19 core intent categories (`ALARM`, `BRIGHTNESS`, `CALL`, `COLOR`, `CREATE_REMINDER`, `LIGHT_OFF`, `LIGHT_ON`, `LIST_REMINDERS`, `MESSAGE`, `NEXT`, `PAUSE`, `PLAY_MUSIC`, `STOP`, `TEMPERATURE`, `TIME`, `TIMER`, `VOLUME_DOWN`, `VOLUME_UP`, `WEATHER`) and associated slot values.
  - 1 explicit `OUT_OF_SCOPE` class (index 93) capturing background noise, non-command speech, and mic ambient room noise.
- **Hierarchical Projection**: Direct argmax yields command variation; a deterministic projection maps variation indices directly to 19 schema intents and slot arguments without multi-stage error cascades.

---

## Training Data & Disjoint Splits
- **Source**: [`airimonda/ai231-me2-voice-commands`](https://huggingface.co/datasets/airimonda/ai231-me2-voice-commands)
- **Pinned Hugging Face Revision**: `6947f13073e57eb6ae67e7e2fc3680700b82aa13`
- **Split Breakdown**:
  - **Train**: 10,222 utterances (5.77 hours, 273 speakers). Includes 608 real Filipino group speech clips across 5 speakers (4x oversampled, 19.03% effective epoch share), 245 out-of-scope speech clips (4x oversampled, class weight 5.0), and 375 ambient room noise slices from physical G-Mark USB microphone.
  - **Validation**: 886 utterances (0.55 hours, 36 speakers). Carved out strictly from the training pool grouped by speaker ID.
  - **Cross-Validation**: 5-Fold Leave-One-Speaker-Out (LOSO) cross-validation across the 5 Filipino training speakers (`202322013`, `202322013_speaker2`, `S1`, `S2`, `S3`).
  - **Test**: 4,443 utterances (2.45 hours, 115 speakers). Completely unseen speakers. 189 real Filipino clips, 76 out-of-scope clips, 635 open-source clips, 3,619 synthetic clips.
  - **Holdout**: 202 utterances (0.18 hours, 5 speakers). Evaluated exclusively on physical hardware.
- **Data Licensing**:
  - Google Speech Commands v2: CC BY 4.0
  - SLURP & MSVCD: CC BY 4.0
  - Mozilla Common Voice v19 & G-Mark Mic Noise: CC0 1.0 Public Domain
  - Fluent Speech Commands: Non-Commercial Research & Education
  - Filipino Group Recordings: AI 231 Coursework Agreement

---

## Architecture & Computational Complexity (BC-ResNet-1)
- **Architecture**: BC-ResNet-1 (base channel factor $c = 32$)
- **Parameters**: **75,710 parameters** (~0.0757 M)
- **Precision**: FP32 (`bcresnet_94class_fp32.onnx`) and INT8 dynamic quantized (`bcresnet_94class_int8.onnx`)
- **Computational Cost** (profiled via `vcmbench/flops.py` on 2.0s input):
  - Multiply-Accumulate Operations (MACs): **42.1 M** (2.35x lighter than DS-CNN baseline at 99.2 M)
  - Floating Point Operations (FLOPs): **86.8 M** (FP32) / **89.6 M** (INT8)
- **Weights Size**:
  - FP32 ONNX: **0.296 MB** (303 KB)
  - INT8 ONNX: **0.117 MB** (120 KB, 2.53x compression)

---

## Inference Rejection Pipeline & Operating Points
To prevent misfires on background speech, conversation, and acoustic transients, inference enforces a dual-stage decision rule:

$$\text{Decision}(x) = \begin{cases} \text{REJECT}, & \text{if } \operatorname{argmax}_c P(c \mid x) = c_{\mathrm{OOS}} \;\lor\; \max_c P(c \mid x) < \tau \\ \operatorname{argmax}_{c < 93} P(c \mid x), & \text{otherwise} \end{cases}$$

where $c_{\mathrm{OOS}} = 93$ (`OUT_OF_SCOPE`).

Threshold $\tau$ was tuned on pooled held-out predictions from 5-fold Leave-One-Speaker-Out (LOSO) cross-validation across all 608 Filipino training clips. Two operating points are reported:
1. **Strict Operating Point ($\tau^* = 0.65$)**: Enforces $\text{FAR}_{\text{OOS}} \le 5.0\%$ on validation OOS speech, achieving high accuracy on accepted commands ($53.85\% \pm 10.29\%$) at the cost of high real-speech rejection ($98.24\% \pm 1.80\%$).
2. **Balanced Operating Point ($\tau_{\text{bal}} = 0.25$)**: Significantly reduces false rejection on real Filipino speech to $64.90\% \pm 5.40\%$, enabling responsive interactive voice assistant usage.

---

## Limitations & Ethical Considerations
- **Synthetic Speech Bias**: Over 75% of training utterances are synthetic text-to-speech recordings. The model achieves higher confidence on synthetic speech than natural expressive speech.
- **Accented & Natural Paraphrasing**: The model is trained on exact variation phrases; out-of-distribution paraphrases or spontaneous disfluencies may be falsely rejected or misclassified.
- **Physical Microphone Matching**: Variations in microphone frequency responses, room acoustics, and SNR will influence rejection calibration.
