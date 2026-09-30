# Machine Exercise 2: Tiny Voice Command Model & Personalized Wake Word Spotter

**Course**: AI 231: Advanced Deep Learning  
**Student**: Misael Maningo  
**Target Hardware**: Raspberry Pi 5 (4-core ARM Cortex-A76)  
**Microphone / Speaker**: G-Mark Micro Go USB (USB Audio Class 1.0)  

---

## 📌 Executive Summary

This project implements a complete, production-grade, low-latency Voice Assistant pipeline deployed on a Raspberry Pi 5 without PyTorch runtime dependencies. The system runs **100% on-device** using pure NumPy feature extraction and ONNX Runtime CPU.

The pipeline comprises two tiny neural networks:
1. **Personalized Wake Word Spotter ("Hey Raspberry")**: A lightweight depthwise-separable CNN (**27.4 KB INT8**) customized to the user's voice and acoustic environment, achieving **98.5% detection confidence** while maintaining **0.00% false-trigger rate** on microphone idle noise.
2. **32-Class Joint Intent & Slot Voice Command Model (VCM)**: A Broadcasted Residual Network (**BC-ResNet-1**, 69,696 parameters, **113.6 KB INT8**) recognizing 13 fixed smart-home intents, 18 parameter-slotted commands, and 1 background rejection class with **99.22% top-1 test accuracy**, **0.9924 Macro F1**, and **15.75 ms CPU inference latency**.

```
[ G-Mark USB Mic ] ──▶ [ Real-Time RMS / VAD Gate ]
                               │ (energy > 0.012)
                               ▼
            [ Streaming Wake Word Spotter: "Hey Raspberry" ]
                               │ (score >= 0.35)
                               ▼
                  ⚡ [ WAKE DETECTED! ]
                               │
            [ Capture 2.0s Audio ──▶ Adaptive Peak Normalizer ]
                               │
              [ Pure NumPy Mel-Spectrogram (40 mels) ]
                               │
           [ 32-Class Joint Intent & Slot BC-ResNet INT8 ]
                               │ (15.75 ms CPU latency)
                               ▼
        ✅ [ ACTION EXECUTED: Intent + Slot Parameter ]
```

---

## 📊 Performance & Benchmark Summary

All models were evaluated on the unseen, speaker-disjoint test set ($10$ unseen speakers, $1,918$ samples) and verified against physical ambient noise collected from the user's G-Mark microphone:

| Model Component | Architecture | Parameters | FP32 Size | INT8 Size | Top-1 Accuracy | Macro F1 | CPU Latency (Single Thread) | Ambient Noise Rejection |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Wake Word Spotter** | `RobustWakeNet` | 23,490 | 46.2 KB | **27.4 KB** | 97.4% (Val) | 0.973 | **~3 ms** | **0.00% false trigger** (100% silent) |
| **Voice Command Model** | `BC-ResNet-1` | 69,696 | 279.2 KB | **113.6 KB** | **99.22%** | **0.9924** | **15.75 ms** | **93.66% avg confidence** (100% rejected) |

### Ambient Noise Rejection Verification (User's G-Mark Mic):
Each of the 15 ambient room noise recordings (`idle_noise_*.wav`) was evaluated using the pure NumPy inference pipeline:
- **15 / 15 takes** correctly classified as `_BACKGROUND_` with **93.66% average confidence** (ranging from $89.77\%$ to $96.03\%$).
- **Digital silence ($0.0$)**: Classified as `_BACKGROUND_` ($70.36\%$).
- **Hesitation / Pauses**: Automatically ignored without false command execution.

---

## 🗂️ Dataset Architecture & Schema

### 1. 32-Class Closed Vocabulary Mapping
The Voice Command Model jointly classifies intent and extracts parameter slots in a single forward pass:

```
├── Fixed Intents (13 Classes):
│   ├── LIGHT_ON          : "turn on the lights", "lights on"
│   ├── LIGHT_OFF         : "turn off the lights", "lights off"
│   ├── PLAY_MUSIC        : "play music", "start playback"
│   ├── PAUSE             : "pause music", "pause playback"
│   ├── STOP              : "stop music", "stop playback"
│   ├── NEXT              : "next song", "skip track"
│   ├── VOLUME_UP         : "turn volume up", "volume up"
│   ├── VOLUME_DOWN       : "turn volume down", "volume down"
│   ├── TIME              : "what time is it", "tell me the time"
│   ├── WEATHER           : "what is the weather", "current weather"
│   ├── CALL              : "call my contact", "make a call"
│   ├── MESSAGE           : "send a message", "dictate message"
│   └── LIST_REMINDERS    : "list my reminders", "show reminders"
│
├── Parameter-Slotted Commands (18 Classes across 6 Slots):
│   ├── TIMER:
│   │   ├── TIMER_10s     : "set timer for 10 seconds" [slot: duration = 10 seconds]
│   │   ├── TIMER_30s     : "set timer for 30 seconds" [slot: duration = 30 seconds]
│   │   └── TIMER_1m      : "set timer for 1 minute"    [slot: duration = 1 minute]
│   ├── ALARM:
│   │   ├── ALARM_6_00AM  : "set alarm for 6 AM"        [slot: time = 6 AM]
│   │   ├── ALARM_8_00AM  : "wake me up at 8 AM"       [slot: time = 8 AM]
│   │   └── ALARM_9_00PM  : "set an alarm for 9 PM"     [slot: time = 9 PM]
│   ├── TEMPERATURE:
│   │   ├── TEMPERATURE_18: "set temperature to 18"    [slot: degrees = 18 degrees]
│   │   ├── TEMPERATURE_22: "thermostat to 22"         [slot: degrees = 22 degrees]
│   │   └── TEMPERATURE_26: "set temperature to 26"    [slot: degrees = 26 degrees]
│   ├── BRIGHTNESS:
│   │   ├── BRIGHTNESS_20 : "brightness to 20 percent" [slot: percent = 20 percent]
│   │   ├── BRIGHTNESS_60 : "brightness to 60 percent" [slot: percent = 60 percent]
│   │   └── BRIGHTNESS_100: "brightness to 100 percent"[slot: percent = 100 percent]
│   ├── COLOR:
│   │   ├── COLOR_RED     : "set light color to red"   [slot: color = red]
│   │   ├── COLOR_BLUE    : "set light color to blue"  [slot: color = blue]
│   │   └── COLOR_GREEN   : "set light color to green" [slot: color = green]
│   └── CREATE_REMINDER:
│       ├── CREATE_REMINDER_DRINK_WATER: "remind me to drink water" [slot: task = drink water]
│       ├── CREATE_REMINDER_STUDY      : "remind me to study"       [slot: task = study]
│       └── CREATE_REMINDER_EXERCISE   : "remind me to exercise"    [slot: task = exercise]
│
└── Rejection Class (1 Class):
    └── _BACKGROUND_      : Environmental noise, room silence, mic hiss, non-command speech
```

### 2. Dataset Design & Splitting
- **Option B Dataset**: 17,986 clean and noisy audio files across 100 synthetic speakers.
- **Speaker-Disjoint Partitions**:
  - **Train**: 80 speakers ($15,332$ samples) + 375 G-Mark ambient noise slices.
  - **Validation**: 10 speakers ($1,938$ samples).
  - **Test**: 10 speakers ($1,918$ samples). Zero speaker overlap between train, val, and test.
- **Acoustic Domain Adaptation**:
  - G-Mark Micro Go USB mic ambient noise floor was captured via `record_user_wakeword.py --noise` ($15$ takes).
  - During GPU training, dynamic additive noise mixing was applied:
    $$\mathbf{x}_{\text{batch}} = \mathbf{x}_{\text{speech}} + \beta \cdot \mathbf{x}_{\text{ambient\_noise}}, \quad \beta \sim \text{Uniform}(0.2, 0.8)$$
  - This bridges the gap between clean synthetic voices and live physical microphone recording.

---

## 🧠 Neural Network Architectures

### 1. RobustWakeNet (Personalized Wake Word Spotter)
- **Input**: $(1, 40, 101)$ Log Mel-Spectrogram ($1.0\text{ s}$ @ $16\text{ kHz}$).
- **Structure**: Initial Conv2d ($3\times3$, stride 2) $\to$ 3 Depthwise-Separable Blocks ($32 \to 48 \to 64$ channels) with BatchNorm, ReLU, Dropout ($0.1$) $\to$ Global Average Pooling $\to$ Linear ($64 \to 2$).
- **Quantization**: Dynamic INT8 quantization via ONNX Runtime (`QuantType.QInt8`), compressed to **27.4 KB**.

### 2. BC-ResNet-1 (Broadcasted Residual Network for VCM)
- **Input**: $(1, 40, 201)$ Log Mel-Spectrogram ($2.0\text{ s}$ @ $16\text{ kHz}$).
- **Structure**: Initial Conv2d ($5\times5$, stride $(2, 1)$, 32 channels) $\to$ 3 Residual Stages ($32 \to 64 \to 96$ channels).
- **Broadcast Temporal Context**: Each block computes temporal pooling across the frequency dimension using `x.mean(dim=2, keepdim=True)`, processes it through a $1\times1$ convolution, and broadcasts it across all frequency channels.
- **Export Compatibility**: Replaced dynamic `AdaptiveAvgPool2d((1, None))` with native `ReduceMean` along axis 2, enabling standard ONNX export.

---

## ⚡ Edge Pipeline Innovations (`demo_rpi5.py`)

1. **Single Persistent ALSA Stream**:
   - G-Mark USB Audio Class 1.0 device triggers `PaErrorCode -9985 (Device unavailable)` when multiple audio streams open and close sequentially.
   - Fixed by keeping a single `sd.InputStream` open indefinitely; switching between circular ring buffer listening and linear command recording on the fly.
2. **Pure NumPy Feature Extractor (`PureNumpyFeatureExtractor`)**:
   - Zero PyTorch or Librosa dependencies. Computes STFT using `np.lib.stride_tricks.as_strided` and `np.fft.rfft`.
   - Matrix-multiplies power spectra by pre-computed Mel filterbanks (`mel_filters_40.npy`).
3. **Adaptive Peak Normalization**:
   - Live speech audio is normalized to $0.85$ peak: `(cmd_samples / cmd_peak) * 0.85`.
   - Eliminates sensitivity to speaker distance, preventing distant speech from dropping below the background threshold.
4. **VAD Energy Gate**:
   - Suppresses model inference when RMS energy $< 0.012$ (muted microphone or dead room silence).
   - Eliminates CPU waste and prevents false wake triggers in quiet environments.
5. **Real-Time ASCII Audio Meter**:
   - Terminal displays live VU meter with wake confidence feedback:
     ```
     [MIC: ████████░░░░░░░░░░ RMS: 0.045 | Wake:  0.0%]
     ```

---

## 🚀 Deployment Instructions (Raspberry Pi 5)

### Step 1: Install Dependencies
Inside your Raspberry Pi 5 terminal:
```bash
# Create virtual environment if not already present
python3 -m venv ~/AI_231_venv
source ~/AI_231_venv/bin/activate

# Install edge runtime packages (100% PyTorch-free)
pip install numpy onnxruntime sounddevice
```

### Step 2: Deploy Files
From your Raspberry Pi 5, sync the project files:
```bash
# Option A: Sync the deployment archive
rsync -avz --progress <USER>@<HPC_IP>:"/path/to/AI\ 231/ME2\ -\ Voice\ Command\ Model/rpi5_slot_deployment.tar.gz" ~/
tar -xzvf ~/rpi5_slot_deployment.tar.gz -C ~/

# Option B: Or clone the git repository
git clone https://github.com/MisaelManingo/AI-231-Submissions-by-Misael.git
cd "AI-231-Submissions-by-Misael/ME2 - Voice Command Model"
```

### Step 3: Run Live Demo
```bash
source ~/AI_231_venv/bin/activate
python demo_rpi5.py
```

Optional CLI flags:
```bash
python demo_rpi5.py --wake_thresh 0.35 --vcm_thresh 0.65 --device <MIC_ID> --samplerate 48000
```

---

## 🛠️ DEVELOPER GUIDE: Implementing Real Actions in Next Phase

> [!IMPORTANT]
> **ATTENTION NEW CONVERSATION / DEVELOPER**:
> **DO NOT retrain or re-export the models.** The models (`wakeword_int8.onnx` and `bcresnet_32_int8.onnx`) are fully trained, quantized, and benchmarked at **99.22% accuracy**.
> All functional changes for implementing real-world actions are made **exclusively inside `demo_rpi5.py`**.

### 1. Where to Implement Actions
Open [`demo_rpi5.py`](./demo_rpi5.py) and locate the method:
```python
def update_state(self, label):
    # Located around lines 320-420
```

### 2. Available Data at Execution Time
When a command is accepted with confidence $\ge 0.65$, `demo_rpi5.py` calls `self.update_state(top_label)`.
Inside this function, you have access to:
- `label`: The exact 32-class label string (e.g. `'BRIGHTNESS_60'`, `'TIMER_10s'`, `'LIGHT_ON'`).
- `meta = self.slot_meta.get(label, {})`:
  - `meta.get("intent")`: Base intent (e.g. `'BRIGHTNESS'`, `'TIMER'`, `'LIGHT_ON'`).
  - `meta.get("slot")`: Slot name (e.g. `'percent'`, `'duration'`, `'color'`, `'degrees'`).
  - `meta.get("slot_value")`: Extracted value (e.g. `'60 percent'`, `'10 seconds'`, `'blue'`, `'22 degrees'`).
- `self.device_state`: Persistent dictionary holding current smart-home state:
  ```python
  self.device_state = {
      "lights": "OFF",
      "brightness": 100,
      "light_color": "WHITE",
      "thermostat": "22°C",
      "music": "STOPPED",
      "volume": 50,
      "alarm": "NONE",
      "timer": "NONE",
      "reminders": []
  }
  ```

### 3. Action Implementation Recipes

#### A. Smart Home Lights / Relays (Raspberry Pi GPIO)
```python
import gpiod

# Initialize GPIO pin for relay/LED
chip = gpiod.Chip('gpiochip4')
light_line = chip.get_line(17)
light_line.request(consumer="voice_assistant", type=gpiod.LINE_REQ_DIR_OUT)

if label == "LIGHT_ON":
    light_line.set_value(1)
    self.device_state["lights"] = "ON"
elif label == "LIGHT_OFF":
    light_line.set_value(0)
    self.device_state["lights"] = "OFF"
```

#### B. System Audio Volume (`amixer` / ALSA)
```python
import subprocess

if label == "VOLUME_UP":
    self.device_state["volume"] = min(100, self.device_state["volume"] + 10)
    subprocess.run(["amixer", "sset", "Master", f"{self.device_state['volume']}%"])
elif label == "VOLUME_DOWN":
    self.device_state["volume"] = max(0, self.device_state["volume"] - 10)
    subprocess.run(["amixer", "sset", "Master", f"{self.device_state['volume']}%"])
```

#### C. Background Timers (`threading.Timer`)
```python
import threading

def play_buzzer():
    print("⏰ [TIMER EXPIRED]: Beep! Beep! Beep!")
    # os.system("aplay /path/to/alarm.wav")

if label == "TIMER_10s":
    self.device_state["timer"] = "10 seconds"
    t = threading.Timer(10.0, play_buzzer)
    t.start()
elif label == "TIMER_30s":
    t = threading.Timer(30.0, play_buzzer)
    t.start()
```

#### D. Spoken Feedback / Text-to-Speech (TTS)
```python
def speak_response(text):
    # Using lightweight espeak-ng on RPi5
    subprocess.run(["espeak-ng", "-v", "en-us", "-s", "160", text])

# Inside update_state:
if label == "TIME":
    current_time = time.strftime('%I:%M %p')
    speak_response(f"The current time is {current_time}")
elif label == "WEATHER":
    speak_response("The weather today is clear with a temperature of 27 degrees.")
```

---

## 📂 File Structure

```
├── demo_rpi5.py                  # Live on-device streaming assistant (PyTorch-free)
├── fast_train_vcm_32.py          # GPU training script for 32-class VCM
├── fast_train_personalized.py    # GPU training script for personalized wake word
├── assemble_dataset_32.py        # Dataset assembler & slot metadata generator
├── record_user_wakeword.py       # Audio recorder for user wake takes & ambient noise
├── models/
│   ├── bcresnet.py               # Broadcasted Residual Network (BC-ResNet-1)
│   ├── dscnn.py                  # Depthwise-Separable CNN benchmark model
│   └── micro_wakeword.py         # MicroWakeNet architecture
├── exports/                      # Quantized production artifacts
│   ├── wakeword_int8.onnx        # 27.4 KB INT8 Wake Word Model
│   ├── bcresnet_32_int8.onnx     # 113.6 KB INT8 32-Class Voice Command Model
│   ├── labels_32.json            # Class index map & slot metadata
│   ├── mel_filters_40.npy        # 40-channel Mel filterbank weights
│   ├── hann_window_400.npy       # Hann window coefficients
│   └── test_metrics_bcresnet_32.json # Recorded test benchmark metrics
└── README.md                     # Comprehensive technical documentation
```

---

## 📜 Academic Integrity & Citation
This work was conducted for **AI 231 (Machine Exercise 2)** at the University of the Philippines.  
References:
- BC-ResNet: *Broadcasted Residual Learning for Efficient Keyword Spotting* (Kim et al., Interspeech 2021).
- Google Speech Commands Dataset v2 (Warden, 2018).
- ONNX Runtime Dynamic Quantization (Microsoft).
