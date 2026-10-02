# Model Card: Broadcasted Residual Network (BC-ResNet-1) 94-Class Voice Command Model

## Model Details
- **Model Name**: BC-ResNet-1 Voice Command Classifier (94 Classes)
- **Model Version**: `v3_94class`
- **Developed by**: Misael Andre Maningo (AI 231, UP Diliman)
- **Model Type**: Deep 2D Convolutional Neural Network with Depthwise-Separable Dilated Residual Blocks and Broadcasted Temporal Frequency Conditioning
- **License**: MIT License
- **Framework**: PyTorch 2.x (Training & Fine-tuning) / ONNX Runtime CPU (Pure NumPy + ONNX inference on Raspberry Pi 5)
- **Target Hardware**: Embedded Edge Linux Devices (Raspberry Pi 5 Model B 8GB, ARM Cortex-A76)

---

## Intended Use & Scope
- **Primary Use**: Low-latency, PyTorch-free voice command recognition for smart home / edge assistant interfaces.
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
  - **Train**: 10,222 utterances (5.77 hours, 273 speakers). Includes 608 real Filipino group speech clips (4x oversampled, 20.19% effective epoch share), 245 out-of-scope speech clips, and 375 ambient room noise slices from physical G-Mark USB microphone.
  - **Validation**: 886 utterances (0.55 hours, 36 speakers). Carved out strictly from the training pool grouped by speaker ID. Includes 1 held-out Filipino speaker (`202521746`, 72 clips) for unbiased zero-leak rejection threshold tuning.
  - **Test**: 4,443 utterances (2.45 hours, 115 speakers). Completely unseen speakers. 189 real Filipino clips, 76 out-of-scope clips, 635 open-source clips, 3,619 synthetic clips.
  - **Holdout**: 202 utterances (0.18 hours, 5 speakers). Evaluated exclusively on physical hardware.
- **Data Licensing**:
  - Google Speech Commands v2: CC BY 4.0
  - SLURP & MSVCD: CC BY 4.0
  - Mozilla Common Voice v19 & G-Mark Mic Noise: CC0 1.0 Public Domain
  - Fluent Speech Commands: Non-Commercial Research & Education
  - Filipino Group Recordings: AI 231 Coursework Agreement

---

## Architecture & Computational Complexity
- **Architecture**: BC-ResNet-1 (base channel factor $c = 32$)
- **Parameters**: 75,646 parameters (~0.076M)
- **Precision**: FP32 (`bcresnet_94class_fp32.onnx`) and INT8 dynamic quantized (`bcresnet_94class_int8.onnx`)
- **Computational Cost**:
  - Multiply-Accumulate Operations (MACs): ~10.4 M
  - Floating Point Operations (FLOPs): ~20.8 M
- **Weights Size**:
  - FP32 ONNX: ~300 KB
  - INT8 ONNX: ~95 KB

---

## Inference Rejection Pipeline
To prevent misfires on background speech, conversation, and acoustic transients, inference enforces a dual-stage decision rule:
$$\text{Decision}(x) = \begin{cases} \text{REJECT}, & \text{if } \operatorname{argmax}_c P(c \mid x) = c_{\mathrm{OOS}} \;\lor\; \max_c P(c \mid x) < \tau \\ \operatorname{argmax}_{c < 93} P(c \mid x), & \text{otherwise} \end{cases}$$
where $c_{\mathrm{OOS}} = 93$ (`OUT_OF_SCOPE`).
Where $\tau$ is tuned strictly on the validation set subject to dual operational constraints:
1. $\text{FAR}_{\text{OOS}} \le 5.0\%$
2. $\text{FRR}_{\text{Filipino}} \le 30.0\%$

---

## Limitations & Ethical Considerations
- **Synthetic Speech Bias**: A significant portion of training data consists of synthetic text-to-speech recordings. Models exhibit higher confidence on synthetic speech than natural expressive speech.
- **Accented & Natural Paraphrasing**: The model is trained on exact variation phrases; out-of-distribution paraphrases or spontaneous disfluencies may be falsely rejected or misclassified.
- **Physical Microphone Matching**: Variations in microphone frequency responses, room acoustics, and SNR will influence rejection calibration.
