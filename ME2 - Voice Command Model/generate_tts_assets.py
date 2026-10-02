#!/usr/bin/env python3
"""
Offline TTS Asset & Chime Generator for Raspberry Pi 5
100% Offline, PyTorch-Free.
Generates all pre-made WAV clips using local 'espeak' (or 'piper' if present),
plus standard synthetic chimes (chime_wake.wav, chime_end.wav) and fallback Alarm.mp3 / sample music.
"""

import os
import sys
import wave
import shutil
import struct
import subprocess
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TTS_DIR = os.path.join(BASE_DIR, "tts")
MUSIC_DIR = os.path.join(BASE_DIR, "Music")
CHIME_WAKE = os.path.join(BASE_DIR, "chime_wake.wav")
CHIME_END = os.path.join(BASE_DIR, "chime_end.wav")
ALARM_MP3 = os.path.join(BASE_DIR, "Alarm.mp3")

SAMPLE_RATE = 44100


def find_tts_engine():
    """Detects available local TTS engines (piper, espeak-ng, espeak)."""
    if shutil.which("piper"):
        return "piper"
    if shutil.which("espeak-ng"):
        return "espeak-ng"
    if shutil.which("espeak"):
        return "espeak"
    return None


def generate_speech_clip(engine, text, out_wav):
    """Generates a WAV file from text using local engine."""
    if os.path.exists(out_wav) and os.path.getsize(out_wav) > 100:
        return  # Already exists

    os.makedirs(os.path.dirname(out_wav), exist_ok=True)

    if engine in ["espeak", "espeak-ng"]:
        cmd = [engine, "-s", "150", "-p", "50", "-w", out_wav, text]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
    elif engine == "piper":
        cmd = ["piper", "--output_file", out_wav]
        proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        proc.communicate(input=text.encode("utf-8"))
    else:
        write_synth_beep(out_wav, duration=0.4, freq=440.0)


def write_synth_beep(filepath, duration=0.3, freq=600.0, sr=SAMPLE_RATE):
    """Writes a clean synthetic sine tone WAV."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    env = np.ones_like(t)
    fade_len = int(0.02 * sr)
    env[:fade_len] = np.linspace(0, 1, fade_len)
    env[-fade_len:] = np.linspace(1, 0, fade_len)
    signal = (0.5 * np.sin(2 * np.pi * freq * t) * env * 32767).astype(np.int16)

    with wave.open(filepath, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(signal.tobytes())


def generate_chimes():
    """Generates pleasant two-tone chimes for wake and end-of-command."""
    print("Generating audio chimes...")

    sr = SAMPLE_RATE
    d = 0.14
    t = np.linspace(0, d, int(sr * d), endpoint=False)
    env = np.sin(np.pi * np.linspace(0, 1, len(t)))
    tone1 = 0.5 * np.sin(2 * np.pi * 523.25 * t) * env
    tone2 = 0.5 * np.sin(2 * np.pi * 783.99 * t) * env
    gap = np.zeros(int(sr * 0.03))
    wake_audio = np.concatenate([tone1, gap, tone2])
    wake_int16 = (wake_audio * 32767).astype(np.int16)

    with wave.open(CHIME_WAKE, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(wake_int16.tobytes())
    print(f"  [OK] {os.path.basename(CHIME_WAKE)}")

    tone3 = 0.45 * np.sin(2 * np.pi * 783.99 * t) * env
    tone4 = 0.45 * np.sin(2 * np.pi * 659.25 * t) * env
    end_audio = np.concatenate([tone3, gap, tone4])
    end_int16 = (end_audio * 32767).astype(np.int16)

    with wave.open(CHIME_END, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(end_int16.tobytes())
    print(f"  [OK] {os.path.basename(CHIME_END)}")


def generate_alarm_and_sample_music():
    """Generates Alarm.mp3 and a sample music track if they do not exist."""
    os.makedirs(MUSIC_DIR, exist_ok=True)

    if not os.path.exists(ALARM_MP3):
        print("Generating fallback Alarm.mp3...")
        sr = 44100
        d_beep = 0.12
        d_silence = 0.08
        t_b = np.linspace(0, d_beep, int(sr * d_beep), endpoint=False)
        beep = (0.7 * np.sin(2 * np.pi * 880 * t_b)).astype(np.float32)
        silence = np.zeros(int(sr * d_silence), dtype=np.float32)
        pulse = np.concatenate([beep, silence, beep, silence, beep, np.zeros(int(sr * 0.4))])
        alarm_raw = (pulse * 32767).astype(np.int16)

        tmp_wav = os.path.join(BASE_DIR, "tmp_alarm.wav")
        with wave.open(tmp_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(alarm_raw.tobytes())

        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-i", tmp_wav, "-b:a", "128k", ALARM_MP3],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            shutil.copy(tmp_wav, ALARM_MP3)

        if os.path.exists(tmp_wav):
            os.remove(tmp_wav)
        print(f"  [OK] {os.path.basename(ALARM_MP3)}")

    sample_song = os.path.join(MUSIC_DIR, "[01] Ambient Sample.mp3")
    if not os.path.exists(sample_song) and len(os.listdir(MUSIC_DIR)) == 0:
        print("Generating sample track in Music/...")
        sr = 44100
        total_sec = 6.0
        t = np.linspace(0, total_sec, int(sr * total_sec), endpoint=False)
        chord1 = 0.25 * np.sin(2 * np.pi * 261.63 * t) + 0.2 * np.sin(2 * np.pi * 329.63 * t) + 0.2 * np.sin(2 * np.pi * 392.00 * t)
        music_raw = (chord1 * 32767).astype(np.int16)

        tmp_song_wav = os.path.join(BASE_DIR, "tmp_song.wav")
        with wave.open(tmp_song_wav, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(music_raw.tobytes())

        if shutil.which("ffmpeg"):
            subprocess.run(["ffmpeg", "-y", "-i", tmp_song_wav, "-b:a", "128k", sample_song],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            shutil.copy(tmp_song_wav, sample_song)

        if os.path.exists(tmp_song_wav):
            os.remove(tmp_song_wav)
        print(f"  [OK] {os.path.basename(sample_song)}")


def main():
    os.makedirs(TTS_DIR, exist_ok=True)
    engine = find_tts_engine()
    print(f"[TTS GENERATOR]: Using engine: '{engine}'")

    clips = {
        "sorry_didnt_catch": "Sorry, I didn't catch that",
        "reminder_exists": "Reminder already exists",
        "reminder_added": "Reminder added",
        "no_reminders": "You have no reminders",
        "your_reminders_are": "Your active reminders are",
        "rem_drink_water": "Drink water",
        "rem_study": "Study",
        "rem_exercise": "Exercise",
        "light_on": "Lights turned on",
        "light_off": "Lights turned off",
        "bright_20": "Brightness set to twenty percent",
        "bright_60": "Brightness set to sixty percent",
        "bright_100": "Brightness set to one hundred percent",
        "color_red": "Light color changed to red",
        "color_green": "Light color changed to green",
        "color_blue": "Light color changed to blue",
        "music_play": "Playing music",
        "music_pause": "Music paused",
        "music_stop": "Music stopped",
        "music_next": "Skipping to next track",
        "vol_up": "Volume increased",
        "vol_down": "Volume decreased",
        "temp_set_18": "Thermostat set to eighteen degrees",
        "temp_set_22": "Thermostat set to twenty-two degrees",
        "temp_set_26": "Thermostat set to twenty-six degrees",
        "timer_10s": "Timer started for ten seconds",
        "timer_30s": "Timer started for thirty seconds",
        "timer_1m": "Timer started for one minute",
        "alarm_6am": "Alarm set for six AM",
        "alarm_8am": "Alarm set for eight AM",
        "alarm_9pm": "Alarm set for nine PM",
        "calling": "Placing phone call",
        "sending_message": "Sending text message",
        "the_time_is": "The time is",
        "the_weather_is": "The current weather is",
        "temperature_is": "Temperature is",
        "humidity_is": "Humidity is",
        "degrees": "degrees Celsius",
        "percent": "percent",
        "am": "AM",
        "pm": "PM",
        "dht_error": "Unable to read weather sensor",
        "cond_hot_humid": "hot and humid",
        "cond_hot_dry": "hot and dry",
        "cond_cool_humid": "cool and humid",
        "cond_cool_dry": "cool and dry",
        "cond_comfortable": "comfortable"
    }

    for n in range(60):
        clips[f"num_{n}"] = str(n)

    print(f"Generating {len(clips)} TTS WAV clips in {TTS_DIR}...")
    for filename_stem, text in clips.items():
        out_wav = os.path.join(TTS_DIR, f"{filename_stem}.wav")
        generate_speech_clip(engine, text, out_wav)

    print(f"[OK] Generated {len(clips)} TTS audio clips.")

    generate_chimes()
    generate_alarm_and_sample_music()

    print("\n✅ All audio assets ready!")


if __name__ == "__main__":
    main()
