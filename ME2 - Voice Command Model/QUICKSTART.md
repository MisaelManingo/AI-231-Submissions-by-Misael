# ME2 Voice Command Assistant: Quick-Start Guide

Real-time, on-device voice assistant pipeline designed for the **Raspberry Pi 5** (ARM Cortex-A76). Completely **PyTorch-free**, optimized for embedded CPU execution via ONNX Runtime and pure NumPy.

---

## 1. Prerequisites

### Hardware
* **Device**: Raspberry Pi 5 (4GB or 8GB recommended; also runs on any Linux, macOS, or Windows computer).
* **Microphone**: USB microphone (tested with G-Mark USB studio mic).
* **Audio Output**: Speaker or 3.5mm / USB DAC (optional, for spoken audio replies).
* **Operating System**: Raspberry Pi OS (64-bit, Debian Bookworm recommended).

---

## 2. Installation & Quick-Start

### Step 1: Extract Package
```bash
unzip ME2-quickstart.zip -d me2_quickstart
cd me2_quickstart
```

### Step 2: Create a Clean Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements-pi.txt
```

### Step 3: Run the No-Hardware Smoke Test
Verify that models, feature extraction, and runtime dependencies load properly without needing a microphone:
```bash
python simulate_demo.py
```
Expected output:
```text
===========================================================================
🧪 RUNNING NO-HARDWARE SIMULATION & BENCHMARK SMOKE TEST (simulate_demo.py)
===========================================================================
Loading ONNX models...
  Wake Model: .../wakeword_int8.onnx
  VCM Model:  .../bcresnet_32class_int8.onnx
  Labels: Loaded 32 classes from .../labels_32.json
...
===========================================================================
✅ SMOKE TEST PASSED: Pure NumPy + ONNX Runtime pipeline is 100% operational!
===========================================================================
```

### Step 4: Run the Live Voice Assistant
Plug in your USB microphone and run:
```bash
python demo_rpi5.py
```
Say `"Hey Raspberry"` to activate the assistant, followed by your command (e.g., `"Lights on"`, `"Timer 30 seconds"`, `"What's the weather?"`).

---

## 3. Package Structure
```text
ME2-quickstart/
├── demo_rpi5.py                  # Live microphone streaming assistant
├── simulate_demo.py              # Zero-hardware simulation smoke test
├── requirements-pi.txt           # Minimal PyTorch-free dependencies
├── QUICKSTART.md                 # This guide
├── data/
│   └── labels_32.json            # 32-class label & slot definition map
└── exports/                      # Quantized INT8 ONNX models and assets
    ├── wakeword_int8.onnx        # MicroWakeNet wake word detector (27.4 KB)
    ├── bcresnet_32class_int8.onnx# BC-ResNet-1 32-class model (113.6 KB)
    ├── labels_32.json            # 32-class label & slot map
    ├── mel_filters_40.npy        # 40-channel Mel filterbanks
    └── hann_window_400.npy       # Hann analysis window
```

---

## 4. Class Benchmark Execution (`airimonda/vcm-benchmark`)
To run the automated benchmark on physical Raspberry Pi hardware from your laptop:
```bash
git clone https://github.com/airimonda/vcm-benchmark.git
cd vcm-benchmark
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python benchmark.py
```
For every recognized command, `demo_rpi5.py` emits the standardized JSON line format:
```json
{"intent": "TIMER", "slot": "30 seconds", "infer_ms": 22.13, "audio_ms": 2000}
```
If rejected or out of scope:
```json
{"intent": "OUT_OF_SCOPE", "slot": null, "infer_ms": 17.13, "audio_ms": 2000}
```
