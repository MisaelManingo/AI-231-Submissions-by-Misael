#!/usr/bin/env python3
"""
Interactive Wake Word Recording Tool for Raspberry Pi 5
Records 10 takes of the user's natural voice saying 'Hey Raspberry'
using their connected USB microphone.
"""

import os
import sys
import time
import wave
import numpy as np

try:
    import sounddevice as sd
except ImportError:
    print("[ERROR]: 'sounddevice' not found. Activate your virtual environment:")
    print("    source ~/AI_231_venv/bin/activate")
    sys.exit(1)

OUTPUT_DIR = "data/my_wakeword"
NUM_TAKES = 10
RECORD_SECONDS = 1.5
TARGET_SR = 16000


def detect_supported_samplerate(device_id=None):
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
            sd.check_input_settings(device=device_id, channels=1, dtype="float32", samplerate=sr)
            return sr
        except Exception:
            continue
    return 48000


def resample_to_16k(audio, orig_sr):
    if orig_sr == TARGET_SR:
        return audio
    num_samples = int(len(audio) * TARGET_SR / orig_sr)
    orig_indices = np.arange(len(audio))
    target_indices = np.linspace(0, len(audio) - 1, num_samples)
    return np.interp(target_indices, orig_indices, audio).astype(np.float32)


def record_idle_noise(hw_sr):
    noise_dir = "data/my_wakeword"
    os.makedirs(noise_dir, exist_ok=True)
    duration = 15.0
    print("=" * 65)
    print("   🤫 IDLE BACKGROUND NOISE CALIBRATION (15 Seconds)")
    print("=" * 65)
    print("Please do NOT speak. Stay silent in your normal room environment.")
    print("This teaches the model your microphone's specific idle background noise.\n")
    input("Press [ENTER] to start recording 15s of silence/ambient noise...")
    
    print("\n🔴 RECORDING IDLE AMBIENT NOISE (15 seconds)...")
    for sec in range(int(duration), 0, -1):
        print(f"Time remaining: {sec:2d}s | Recording room noise...", end="\r")
        time.sleep(1.0)
    print("\nCapturing stream from device...")
    
    audio_data = sd.rec(int(duration * hw_sr), samplerate=hw_sr, channels=1, dtype="float32")
    sd.wait()
    
    samples_16k = resample_to_16k(audio_data[:, 0], hw_sr)
    rms = float(np.sqrt(np.mean(samples_16k ** 2)))
    print(f"✅ Captured! Average Idle RMS: {rms:.4f}")
    
    # Slice into 15 1.0-second negative clips
    for i in range(15):
        chunk = samples_16k[i * 16000 : (i + 1) * 16000]
        if len(chunk) < 16000:
            break
        int16_chunk = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16)
        wav_path = os.path.join(noise_dir, f"idle_noise_{i+1:02d}.wav")
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SR)
            wf.writeframes(int16_chunk.tobytes())
            
    print(f"🎉 Saved 15 idle background noise clips to '{noise_dir}/idle_noise_*.wav'!")
    print("=" * 65)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Wake Word & Idle Noise Recorder for RPi5")
    parser.add_argument("--noise", action="store_true", help="Record 15s of idle mic noise instead of wake word")
    args = parser.parse_args()

    hw_sr = detect_supported_samplerate()
    dev_name = sd.query_devices(kind="input")["name"]
    print(f"Microphone: '{dev_name}' | Hardware Rate: {hw_sr} Hz\n")
    
    if args.noise:
        record_idle_noise(hw_sr)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print("=" * 65)
    print("   🎙️  WAKE WORD RECORDER: 'Hey Raspberry' (10 Short Takes)")
    print("=" * 65)
    print("This will record 10 short 1.5-second clips of your natural voice.")
    print("Vary your tone slightly (e.g. normal, slightly fast, distant, casual).\n")
    input("Press [ENTER] to start recording...")
    
    for i in range(1, NUM_TAKES + 1):
        print(f"\n--- [Take {i}/{NUM_TAKES}] ---")
        for count in [3, 2, 1]:
            print(f"Get ready: {count}...", end="\r")
            time.sleep(0.6)
            
        print("🔴 SPEAK NOW: 'Hey Raspberry'!              ")
        audio_data = sd.rec(int(RECORD_SECONDS * hw_sr), samplerate=hw_sr, channels=1, dtype="float32")
        sd.wait()
        
        samples_16k = resample_to_16k(audio_data[:, 0], hw_sr)
        rms = float(np.sqrt(np.mean(samples_16k ** 2)))
        peak = float(np.max(np.abs(samples_16k)))
        
        if peak < 0.02:
            print(f"⚠️  Very quiet (RMS: {rms:.3f}). Make sure your mic is unmuted!")
        else:
            print(f"✅ Captured! (RMS: {rms:.3f}, Peak: {peak:.2f})")
            
        wav_path = os.path.join(OUTPUT_DIR, f"user_wake_{i:02d}.wav")
        int16_samples = (np.clip(samples_16k, -1.0, 1.0) * 32767.0).astype(np.int16)
        with wave.open(wav_path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(TARGET_SR)
            wf.writeframes(int16_samples.tobytes())
            
        time.sleep(0.5)

    print("\n" + "=" * 65)
    print(f"🎉 All {NUM_TAKES} takes recorded successfully in '{OUTPUT_DIR}/'!")
    print("=" * 65)


if __name__ == "__main__":
    main()
