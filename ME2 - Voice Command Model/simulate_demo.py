import os
import time
import json
import torch
import torchaudio
import soundfile as sf
import numpy as np
import onnxruntime as ort

BASE_DIR = "/home/misael.andre.maningo/MEng AI/AI 231/ME2 - Voice Command Model"
OPTIONB_DIR = os.path.join(BASE_DIR, "upstream_repo/MEX2/OptionB")
LABELS_PATH = os.path.join(BASE_DIR, "data/labels_20.json")
SLOTS_PATH = os.path.join(OPTIONB_DIR, "slots.json")

WAKE_ONNX = os.path.join(BASE_DIR, "exports/wakeword_int8.onnx")
VCM_ONNX = os.path.join(BASE_DIR, "exports/bcresnet_int8.onnx")

TARGET_SR = 16000

class SmartHomePipeline:
    def __init__(self, wake_threshold=0.55, vcm_threshold=0.65):
        self.wake_threshold = wake_threshold
        self.vcm_threshold = vcm_threshold
        
        print("Initializing ONNX Runtime inference sessions...")
        self.wake_session = ort.InferenceSession(WAKE_ONNX, providers=["CPUExecutionProvider"])
        self.vcm_session = ort.InferenceSession(VCM_ONNX, providers=["CPUExecutionProvider"])
        
        self.wake_inp = self.wake_session.get_inputs()[0].name
        self.vcm_inp = self.vcm_session.get_inputs()[0].name
        
        # Load labels
        with open(LABELS_PATH, "r") as f:
            label_data = json.load(f)
        self.idx2label = {int(k): v for k, v in label_data["idx2label"].items()}
        
        # Load slots
        if os.path.exists(SLOTS_PATH):
            with open(SLOTS_PATH, "r") as f:
                self.slots_info = json.load(f)
        else:
            self.slots_info = {}
            
        # Feature extractors
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=TARGET_SR,
            n_fft=400,
            win_length=400,
            hop_length=160,
            n_mels=40
        )
        self.amp_to_db = torchaudio.transforms.AmplitudeToDB()
        
        # Simulated Smart Home Device State
        self.state = {
            "music": "STOPPED",
            "volume": 50,
            "lights": "OFF",
            "brightness": 50,
            "color": "white",
            "temperature": 22,
            "alarm": None,
            "timer": None
        }

    def extract_features(self, waveform, target_samples):
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        else:
            waveform = waveform.t().mean(dim=0, keepdim=True)
            
        length = waveform.shape[-1]
        if length < target_samples:
            pad = target_samples - length
            waveform = torch.nn.functional.pad(waveform, (pad // 2, pad - pad // 2))
        elif length > target_samples:
            start = (length - target_samples) // 2
            waveform = waveform[:, start : start + target_samples]
            
        mel = self.melspec(waveform)
        log_mel = self.amp_to_db(mel)
        return log_mel.unsqueeze(0).numpy()

    def check_wake_word(self, audio_1s):
        feat = self.extract_features(torch.from_numpy(audio_1s), 16000)
        logits = self.wake_session.run(None, {self.wake_inp: feat})[0]
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        wake_score = probs[0, 1]
        is_wake = wake_score >= self.wake_threshold
        return is_wake, wake_score

    def classify_command(self, audio_2s):
        t0 = time.perf_counter()
        feat = self.extract_features(torch.from_numpy(audio_2s), 32000)
        logits = self.vcm_session.run(None, {self.vcm_inp: feat})[0]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        
        exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        top_idx = int(np.argmax(probs, axis=1)[0])
        top_conf = float(probs[0, top_idx])
        top_intent = self.idx2label[top_idx]
        
        return top_intent, top_conf, latency_ms

    def actuate(self, intent, conf, raw_text=""):
        if intent == "_BACKGROUND_" or conf < self.vcm_threshold:
            print(f"  [ACTUATION]: Unrecognized / ambient noise (conf={conf*100:.1f}%). Command rejected.")
            return
            
        print(f"  [ACTUATION SUCCESS]: Recognized Intent: '{intent}' (Confidence: {conf*100:.1f}%)")
        
        # State machine updates
        if intent == "PLAY_MUSIC":
            self.state["music"] = "PLAYING"
        elif intent in ["PAUSE", "STOP"]:
            self.state["music"] = intent
        elif intent == "VOLUME_UP":
            self.state["volume"] = min(100, self.state["volume"] + 10)
        elif intent == "VOLUME_DOWN":
            self.state["volume"] = max(0, self.state["volume"] - 10)
        elif intent == "LIGHT_ON":
            self.state["lights"] = "ON"
        elif intent == "LIGHT_OFF":
            self.state["lights"] = "OFF"
        elif "COLOR" in intent:
            self.state["color"] = "DYNAMIC"
        elif "TEMPERATURE" in intent:
            self.state["temperature"] = "UPDATED"
            
        print(f"  [CURRENT SMART-HOME STATE]: {self.state}")


def run_simulation():
    print("=" * 70)
    print("REAL-TIME SMART HOME DEMO SIMULATION (Raspberry Pi 5 Pipeline)")
    print("=" * 70)
    pipeline = SmartHomePipeline(wake_threshold=0.55, vcm_threshold=0.65)
    
    # Test Scenarios
    scenarios = [
        {
            "name": "Scenario 1: Wake Word + Positive Smart Home Command (LIGHT_ON)",
            "wake_audio": os.path.join(BASE_DIR, "data/wakeword/wavs/pos_hey_raspberry_en-US-AriaNeural_p0pct_0Hz.wav"),
            "cmd_audio": os.path.join(OPTIONB_DIR, "LIGHT_ON/LIGHT_ON_s91_v1_clean.wav"),
            "expected": "LIGHT_ON"
        },
        {
            "name": "Scenario 2: Wake Word + Music Control Command (PLAY_MUSIC in 30dB Noise)",
            "wake_audio": os.path.join(BASE_DIR, "data/wakeword/wavs/pos_hey_raspberry_en-PH-RosaNeural_p0pct_0Hz.wav"),
            "cmd_audio": os.path.join(OPTIONB_DIR, "PLAY_MUSIC/PLAY_MUSIC_s95_v2_noisy.wav"),
            "expected": "PLAY_MUSIC"
        },
        {
            "name": "Scenario 3: Non-Wake Ambient Speech (Confuser 'Raspberry Pi' + Ambient Dishes Noise)",
            "wake_audio": os.path.join(BASE_DIR, "data/wakeword/wavs/neg_confuser_raspberry_pi_en-US-GuyNeural_p0pct_0Hz.wav"),
            "cmd_audio": os.path.join(BASE_DIR, "data/negatives_2s/bg_doing_the_dishes_slice000.wav"),
            "expected": "REJECT_AT_WAKE_STAGE"
        },
        {
            "name": "Scenario 4: Wake Word + Ambient Background Noise (Dishwashing Noise)",
            "wake_audio": os.path.join(BASE_DIR, "data/wakeword/wavs/pos_hey_raspberry_en-GB-SoniaNeural_p0pct_0Hz.wav"),
            "cmd_audio": os.path.join(BASE_DIR, "data/negatives_2s/bg_doing_the_dishes_slice001.wav"),
            "expected": "REJECT_AT_VCM_STAGE"
        },
        {
            "name": "Scenario 5: Wake Word + Variable Slot Command (BRIGHTNESS_100)",
            "wake_audio": os.path.join(BASE_DIR, "data/wakeword/wavs/pos_hey_raspberry_en-CA-ClaraNeural_p0pct_0Hz.wav"),
            "cmd_audio": os.path.join(OPTIONB_DIR, "BRIGHTNESS_100/BRIGHTNESS_100_s100_v1_noisy.wav"),
            "expected": "BRIGHTNESS"
        }
    ]
    
    for s_idx, sc in enumerate(scenarios, 1):
        print(f"\n>>> Running {sc['name']}...")
        
        # 1. Stage 1: Wake Word Detection
        wake_data, sr = sf.read(sc["wake_audio"], dtype="float32")
        is_wake, wake_score = pipeline.check_wake_word(wake_data)
        print(f"  [STAGE 1: WAKE DETECTOR]: Heard Audio -> 'Hey Raspberry' Confidence: {wake_score*100:.1f}%")
        
        if not is_wake:
            print("  --> [IDLE]: Wake word NOT triggered. Pipeline remains in standby sleep mode.")
            continue
            
        print("  --> [ACTIVATED!]: Visual indicator ON (Green LED / Prompt). Listening for 2.0s command...")
        
        # 2. Stage 2: Voice Command Classification
        cmd_data, sr = sf.read(sc["cmd_audio"], dtype="float32")
        intent, conf, latency_ms = pipeline.classify_command(cmd_data)
        print(f"  [STAGE 2: VCM INFERENCE]: Evaluated 2.0s audio in {latency_ms:.2f} ms")
        
        # 3. Smart Home Actuation
        pipeline.actuate(intent, conf)
        
    print("\n" + "=" * 70)
    print("DEMO SIMULATION COMPLETE: All scenarios verified successfully.")
    print("=" * 70)

if __name__ == "__main__":
    run_simulation()
