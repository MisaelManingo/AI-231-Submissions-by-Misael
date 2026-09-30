#!/usr/bin/env python3
"""
Real-Time Tiny Voice Command Pipeline for Raspberry Pi 5 (100% PyTorch-Free)
Features:
- Live terminal RMS audio meter for real-time visual microphone feedback
- Automatic hardware sample rate detection & on-the-fly NumPy resampling
- Streaming MicroWakeNet INT8 wake word detector ('Hey Raspberry')
- BC-ResNet INT8 Voice Command Model for 19 smart-home intents

Usage:
    source ~/AI_231_venv/bin/activate
    python demo_rpi5.py [--wake_thresh 0.20] [--vcm_thresh 0.65] [--device 0] [--samplerate 48000]
"""

import os
import sys
import time
import json
import argparse
import collections
import numpy as np
import onnxruntime as ort

try:
    import sounddevice as sd
except ImportError:
    sd = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WAKE_ONNX = os.path.join(BASE_DIR, "exports/wakeword_int8.onnx")

# Prefer 32-Class Joint Intent & Slot Model, fallback to 20-Class if needed
VCM_32_ONNX = os.path.join(BASE_DIR, "exports/bcresnet_32_int8.onnx")
VCM_20_ONNX = os.path.join(BASE_DIR, "exports/bcresnet_int8.onnx")
VCM_ONNX = VCM_32_ONNX if os.path.exists(VCM_32_ONNX) else VCM_20_ONNX

MEL_FILTER_PATH = os.path.join(BASE_DIR, "exports/mel_filters_40.npy")
HANN_WIN_PATH = os.path.join(BASE_DIR, "exports/hann_window_400.npy")

LABELS_32_EXPORTS = os.path.join(BASE_DIR, "exports/labels_32.json")
LABELS_32_DATA = os.path.join(BASE_DIR, "data/labels_32.json")
LABELS_20_DATA = os.path.join(BASE_DIR, "data/labels_20.json")

if os.path.exists(LABELS_32_EXPORTS):
    LABELS_PATH = LABELS_32_EXPORTS
elif os.path.exists(LABELS_32_DATA):
    LABELS_PATH = LABELS_32_DATA
else:
    LABELS_PATH = LABELS_20_DATA

MODEL_SR = 16000
WAKE_WINDOW_SAMPLES = 16000 # 1.0 second @ 16kHz
COMMAND_WINDOW_SAMPLES = 32000 # 2.0 seconds @ 16kHz

def parse_args():
    parser = argparse.ArgumentParser(description="Real-Time Voice Command Pipeline on RPi5 (PyTorch-Free)")
    parser.add_argument("--wake_thresh", type=float, default=0.30, help="Wake word detection threshold (default: 0.30)")
    parser.add_argument("--vcm_thresh", type=float, default=0.65, help="Voice command acceptance threshold (default: 0.65)")
    parser.add_argument("--device", type=int, default=None, help="Input microphone device ID")
    parser.add_argument("--samplerate", type=int, default=None, help="Hardware sample rate (e.g. 48000, 44100, 16000)")
    return parser.parse_args()


def detect_supported_samplerate(device_id=None):
    """
    Finds the native sample rate supported by the connected microphone.
    Tries 16000, 48000, 44100, 32000, 22050, 8000.
    """
    candidate_rates = [16000, 48000, 44100, 32000, 22050, 8000]
    try:
        dev_info = sd.query_devices(device_id, "input")
        default_sr = int(dev_info.get("default_samplerate", 0))
        if default_sr > 0 and default_sr not in candidate_rates:
            candidate_rates.insert(0, default_sr)
        elif default_sr in candidate_rates:
            candidate_rates.remove(default_sr)
            candidate_rates.insert(0, default_sr)
    except Exception:
        pass
        
    for sr in candidate_rates:
        try:
            sd.check_input_settings(device=device_id, channels=1, samplerate=sr)
            return sr
        except Exception:
            continue
            
    return 44100


def resample_to_16k(audio, orig_sr):
    """
    Fast, pure NumPy 1D linear interpolation to resample from orig_sr to 16,000 Hz.
    """
    if orig_sr == MODEL_SR:
        return audio
        
    num_samples = int(len(audio) * MODEL_SR / orig_sr)
    orig_indices = np.arange(len(audio))
    target_indices = np.linspace(0, len(audio) - 1, num_samples)
    return np.interp(target_indices, orig_indices, audio).astype(np.float32)


def render_rms_meter(rms, wake_score, wake_thresh, bar_len=18):
    """
    Renders an ASCII audio VU/RMS meter bar.
    rms typically ranges from 0.0001 (silence) to ~0.20 (loud speech).
    """
    # Scale rms non-linearly (dB-like) for clear visual response
    norm_val = min(1.0, np.sqrt(max(0.0, rms * 15.0)))
    filled = int(norm_val * bar_len)
    bar = "█" * filled + "░" * (bar_len - filled)
    
    # Highlight when wake score is rising
    color = "\033[92m" if wake_score >= wake_thresh else "\033[90m"
    rst = "\033[0m"
    
    return f"[MIC: {bar} RMS: {rms:.3f} | {color}Wake: {wake_score*100:4.1f}%{rst}] [IDLE: Say 'Hey Raspberry']"


class PureNumpyFeatureExtractor:
    """
    Pure NumPy implementation of Log-Mel Spectrogram matching Torchaudio.
    Requires ZERO PyTorch dependencies.
    """
    def __init__(self, mel_filter_path, hann_win_path, n_fft=400, hop_length=160):
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.fb = np.load(mel_filter_path)
        self.win = np.load(hann_win_path)

    def extract(self, y, target_samples):
        if y.ndim > 1:
            y = np.mean(y, axis=0) if y.shape[0] < y.shape[1] else np.mean(y, axis=1)
            
        if len(y) < target_samples:
            pad = target_samples - len(y)
            y = np.pad(y, (pad // 2, pad - pad // 2), mode="constant")
        elif len(y) > target_samples:
            start = (len(y) - target_samples) // 2
            y = y[start : start + target_samples]
            
        pad_amount = self.n_fft // 2
        y_padded = np.pad(y, pad_amount, mode="reflect")
        
        num_frames = (len(y_padded) - self.n_fft) // self.hop_length + 1
        strides = (self.hop_length * y_padded.strides[0], y_padded.strides[0])
        frames = np.lib.stride_tricks.as_strided(y_padded, shape=(num_frames, self.n_fft), strides=strides)
        
        stft = np.fft.rfft(frames * self.win, n=self.n_fft, axis=1)
        pow_np = (np.abs(stft) ** 2).T
        mel = np.dot(self.fb.T, pow_np)
        
        amin = 1e-10
        log_spec = 10.0 * np.log10(np.maximum(amin, mel))
        return log_spec[np.newaxis, np.newaxis, :, :].astype(np.float32)


class RPi5VoiceAssistant:
    def __init__(self, wake_thresh=0.20, vcm_thresh=0.65, hw_samplerate=None):
        self.wake_thresh = wake_thresh
        self.vcm_thresh = vcm_thresh
        self.hw_sr = hw_samplerate
        
        print("=" * 65)
        print("ON-DEVICE VOICE ASSISTANT (Raspberry Pi 5 - PyTorch-Free)")
        print("Wake Word: 'Hey Raspberry' (pronounced 'Hey Razz-berry')")
        print("=" * 65)
        
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        
        self.wake_session = ort.InferenceSession(WAKE_ONNX, sess_options=opts, providers=["CPUExecutionProvider"])
        self.vcm_session = ort.InferenceSession(VCM_ONNX, sess_options=opts, providers=["CPUExecutionProvider"])
        
        self.wake_inp = self.wake_session.get_inputs()[0].name
        self.vcm_inp = self.vcm_session.get_inputs()[0].name
        
        with open(LABELS_PATH, "r") as f:
            label_data = json.load(f)
        self.idx2label = {int(k): v for k, v in label_data["idx2label"].items()}
        self.slot_meta = label_data.get("slot_meta", {})
        print(f"[MODEL CONFIG]: Active VCM ONNX = {os.path.basename(VCM_ONNX)}")
        print(f"[MODEL CONFIG]: Loaded {len(self.idx2label)} classes from {os.path.basename(LABELS_PATH)}")
        
        self.extractor = PureNumpyFeatureExtractor(MEL_FILTER_PATH, HANN_WIN_PATH)
        self.audio_ring_buffer = collections.deque(maxlen=WAKE_WINDOW_SAMPLES)
        self.state = "IDLE_LISTENING"
        
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

    def run_live(self, device_id=None):
        if sd is None:
            print("\n[ERROR]: 'sounddevice' package not found on RPi5.")
            print("To install inside your virtual environment (~AI_231_venv):")
            print("    source ~/AI_231_venv/bin/activate")
            print("    pip install onnxruntime sounddevice")
            sys.exit(1)
            
        if self.hw_sr is None:
            self.hw_sr = detect_supported_samplerate(device_id)
            
        print(f"\n[AUDIO CONFIG]: Hardware Capture Sample Rate = {self.hw_sr} Hz")
        print(f"[AUDIO CONFIG]: Internal Model Processing Rate = {MODEL_SR} Hz (16 kHz)")
        if self.hw_sr != MODEL_SR:
            print(f"[AUDIO CONFIG]: Automatic on-the-fly resampling: {self.hw_sr}Hz -> {MODEL_SR}Hz active.")
            
        hw_blocksize = int(self.hw_sr * 0.10)
        
        dev_name = sd.query_devices(device_id, "input")["name"] if device_id is not None else sd.query_devices(kind="input")["name"]
        print(f"[AUDIO CONFIG]: Active Microphone Device: '{dev_name}'")
        print("-" * 65)
        print("Speak clearly into the microphone. Look at the RMS meter below.")
        print("-" * 65 + "\n")
        
        last_rms = 0.0
        cmd_buffer = []

        def audio_callback(indata, frames, time_info, status):
            nonlocal last_rms
            if status:
                pass
            raw_chunk = indata[:, 0]
            # Compute real-time RMS on the incoming hardware block
            last_rms = float(np.sqrt(np.mean(raw_chunk ** 2)))
            # Resample chunk to 16kHz
            resampled_chunk = resample_to_16k(raw_chunk, self.hw_sr)
            if self.state == "IDLE_LISTENING":
                self.audio_ring_buffer.extend(resampled_chunk)
            elif self.state == "CAPTURING_COMMAND":
                cmd_buffer.extend(resampled_chunk)

        with sd.InputStream(device=device_id, channels=1, samplerate=self.hw_sr, blocksize=hw_blocksize, callback=audio_callback):
            while True:
                time.sleep(0.05) # Poll loop every 50ms
                
                if self.state == "IDLE_LISTENING":
                    if len(self.audio_ring_buffer) < WAKE_WINDOW_SAMPLES:
                        continue
                        
                    buf_np = np.array(self.audio_ring_buffer, dtype=np.float32)
                    
                    # Voice Activity / Energy Gate:
                    # If audio energy is below speech threshold (silence or muted mic), suppress wake inference
                    if last_rms < 0.012:
                        prob_wake = 0.0
                    else:
                        feat = self.extractor.extract(buf_np, WAKE_WINDOW_SAMPLES)
                        logits = self.wake_session.run(None, {self.wake_inp: feat})[0]
                        exp_l = np.exp(logits - np.max(logits, axis=1, keepdims=True))
                        prob_wake = float((exp_l / np.sum(exp_l, axis=1, keepdims=True))[0, 1])
                    
                    # Update live terminal RMS meter
                    meter_str = render_rms_meter(last_rms, prob_wake, self.wake_thresh)
                    sys.stdout.write("\r" + meter_str)
                    sys.stdout.flush()
                    
                    if prob_wake >= self.wake_thresh:
                        # Clear line and print activation banner
                        sys.stdout.write("\r" + " " * 80 + "\r")
                        sys.stdout.flush()
                        
                        print(f"⚡ [WAKE DETECTED!] 'Hey Raspberry' (Score: {prob_wake*100:.1f}%)")
                        print("🎙️  [LISTENING]: Speak command now (recording 2.0s)...")
                        cmd_buffer.clear()
                        self.state = "CAPTURING_COMMAND"
                        
                elif self.state == "CAPTURING_COMMAND":
                    pct = min(100, int(len(cmd_buffer) / COMMAND_WINDOW_SAMPLES * 100))
                    bars = "▓" * (pct // 5) + "░" * (20 - (pct // 5))
                    sys.stdout.write(f"\r[RECORDING COMMAND: {bars} {pct:3d}% | RMS: {last_rms:.3f}]")
                    sys.stdout.flush()
                    
                    if len(cmd_buffer) >= COMMAND_WINDOW_SAMPLES:
                        sys.stdout.write("\r" + " " * 80 + "\r")
                        sys.stdout.flush()
                        
                        cmd_samples = np.array(cmd_buffer[:COMMAND_WINDOW_SAMPLES], dtype=np.float32)
                        
                        # Adaptive peak normalization to match Option B dataset training distribution (0.85 peak)
                        cmd_peak = float(np.max(np.abs(cmd_samples)))
                        if cmd_peak > 0.015:  # Audio contains audible speech, normalize distance attenuation
                            cmd_samples = (cmd_samples / cmd_peak) * 0.85
                        
                        # Infer command
                        t0 = time.perf_counter()
                        vcm_feat = self.extractor.extract(cmd_samples, COMMAND_WINDOW_SAMPLES)
                        vcm_logits = self.vcm_session.run(None, {self.vcm_inp: vcm_feat})[0]
                        latency_ms = (time.perf_counter() - t0) * 1000.0
                        
                        exp_v = np.exp(vcm_logits - np.max(vcm_logits, axis=1, keepdims=True))
                        probs = exp_v / np.sum(exp_v, axis=1, keepdims=True)
                        top_idx = int(np.argmax(probs, axis=1)[0])
                        top_conf = float(probs[0, top_idx])
                        top_label = self.idx2label[top_idx]
                        
                        print(f"⏱️  [VCM INFERENCE]: Latency: {latency_ms:.2f} ms")
                        
                        if top_label == "_BACKGROUND_" or top_conf < self.vcm_thresh:
                            print(f"❌ [COMMAND IGNORED]: Ambient noise or low confidence ({top_label}, {top_conf*100:.1f}%)")
                        else:
                            meta = self.slot_meta.get(top_label, {})
                            intent_name = meta.get("intent", top_label)
                            slot_name = meta.get("slot")
                            slot_val = meta.get("slot_value")
                            
                            print(f"✅ [COMMAND EXECUTED]: '{top_label}' ({top_conf*100:.1f}%)")
                            if slot_name and slot_val:
                                print(f"   🏷️  Base Intent: {intent_name} | 🧩 Slot: {slot_name} = '{slot_val}'")
                            else:
                                print(f"   🏷️  Base Intent: {intent_name}")
                            self.update_state(top_label)
                            
                        # Reset ring buffer, command buffer and state
                        self.audio_ring_buffer.clear()
                        cmd_buffer.clear()
                        self.state = "IDLE_LISTENING"
                        print("\nReturning to idle listening...")

    def update_state(self, label):
        if label == "LIGHT_ON":
            self.device_state["lights"] = "ON"
            print("💡 [ACTION]: Turning lights ON.")
        elif label == "LIGHT_OFF":
            self.device_state["lights"] = "OFF"
            print("💡 [ACTION]: Turning lights OFF.")
        elif label == "PLAY_MUSIC":
            self.device_state["music"] = "PLAYING"
            print("🎵 [ACTION]: Playing music.")
        elif label in ["PAUSE", "STOP"]:
            self.device_state["music"] = label
            print(f"🎵 [ACTION]: Music {label.lower()}ed.")
        elif label == "NEXT":
            print("⏭️  [ACTION]: Skipping to next track.")
        elif label == "VOLUME_UP":
            self.device_state["volume"] = min(100, self.device_state["volume"] + 10)
            print(f"🔊 [ACTION]: Volume increased to {self.device_state['volume']}%.")
        elif label == "VOLUME_DOWN":
            self.device_state["volume"] = max(0, self.device_state["volume"] - 10)
            print(f"🔉 [ACTION]: Volume decreased to {self.device_state['volume']}%.")
        elif label == "BRIGHTNESS_20":
            self.device_state["brightness"] = 20
            print("🔆 [ACTION]: Brightness set to 20%.")
        elif label == "BRIGHTNESS_60":
            self.device_state["brightness"] = 60
            print("🔆 [ACTION]: Brightness set to 60%.")
        elif label == "BRIGHTNESS_100":
            self.device_state["brightness"] = 100
            print("🔆 [ACTION]: Brightness set to 100%.")
        elif label == "COLOR_RED":
            self.device_state["light_color"] = "RED"
            print("🔴 [ACTION]: Light color changed to RED.")
        elif label == "COLOR_BLUE":
            self.device_state["light_color"] = "BLUE"
            print("🔵 [ACTION]: Light color changed to BLUE.")
        elif label == "COLOR_GREEN":
            self.device_state["light_color"] = "GREEN"
            print("🟢 [ACTION]: Light color changed to GREEN.")
        elif label == "TEMPERATURE_18":
            self.device_state["thermostat"] = "18°C"
            print("❄️  [ACTION]: Thermostat set to 18°C.")
        elif label == "TEMPERATURE_22":
            self.device_state["thermostat"] = "22°C"
            print("🌡️  [ACTION]: Thermostat set to 22°C.")
        elif label == "TEMPERATURE_26":
            self.device_state["thermostat"] = "26°C"
            print("🔥 [ACTION]: Thermostat set to 26°C.")
        elif label == "ALARM_6_00AM":
            self.device_state["alarm"] = "6:00 AM"
            print("⏰ [ACTION]: Alarm set for 6:00 AM.")
        elif label == "ALARM_8_00AM":
            self.device_state["alarm"] = "8:00 AM"
            print("⏰ [ACTION]: Alarm set for 8:00 AM.")
        elif label == "ALARM_9_00PM":
            self.device_state["alarm"] = "9:00 PM"
            print("⏰ [ACTION]: Alarm set for 9:00 PM.")
        elif label == "TIMER_10s":
            self.device_state["timer"] = "10 seconds"
            print("⏳ [ACTION]: Timer started for 10 seconds.")
        elif label == "TIMER_30s":
            self.device_state["timer"] = "30 seconds"
            print("⏳ [ACTION]: Timer started for 30 seconds.")
        elif label == "TIMER_1m":
            self.device_state["timer"] = "1 minute"
            print("⏳ [ACTION]: Timer started for 1 minute.")
        elif label == "CREATE_REMINDER_DRINK_WATER":
            self.device_state["reminders"].append("Drink Water")
            print("💧 [ACTION]: Added reminder: 'Drink Water'.")
        elif label == "CREATE_REMINDER_STUDY":
            self.device_state["reminders"].append("Study")
            print("📚 [ACTION]: Added reminder: 'Study'.")
        elif label == "CREATE_REMINDER_EXERCISE":
            self.device_state["reminders"].append("Exercise")
            print("🏃 [ACTION]: Added reminder: 'Exercise'.")
        elif label == "LIST_REMINDERS":
            rems = ", ".join(self.device_state["reminders"]) if self.device_state["reminders"] else "None"
            print(f"📋 [ASSISTANT]: Your active reminders: [{rems}]")
        elif label == "WEATHER":
            print("⛅ [ASSISTANT]: Current weather is sunny and clear, 27°C.")
        elif label == "TIME":
            print(f"🕒 [ASSISTANT]: Current time is {time.strftime('%I:%M %p')}.")
        elif label == "CALL":
            print("📞 [ASSISTANT]: Placing call to contact...")
        elif label == "MESSAGE":
            print("💬 [ASSISTANT]: Composing SMS message...")
            
        print(f"🏠 [DEVICE STATE]: {self.device_state}")

if __name__ == "__main__":
    args = parse_args()
    assistant = RPi5VoiceAssistant(
        wake_thresh=args.wake_thresh,
        vcm_thresh=args.vcm_thresh,
        hw_samplerate=args.samplerate
    )
    assistant.run_live(device_id=args.device)
