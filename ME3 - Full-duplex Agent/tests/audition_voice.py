#!/usr/bin/env python3
"""
Gemini Live Voice Audition Utility
Allows fast command-line auditioning of Gemini Live prebuilt voices:
- Aoede (Breezy, conversational)
- Kore (Firm, crisp, professional)
- Zephyr (Bright, clear)
- Puck (Upbeat, energetic)
- Charon (Deep, calm, authoritative)
- Fenrir (Excitable, high-energy)

Usage:
    python tests/audition_voice.py --voice Kore
    python tests/audition_voice.py --voice Zephyr
    python tests/audition_voice.py --voice Aoede
"""

import sys
import argparse
import asyncio
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import sounddevice as sd
import soundfile as sf
from google import genai
from google.genai import types

from config import GOOGLE_API_KEY, GEMINI_LIVE_MODEL, AUDIO_DEVICE_NAME

SUPPORTED_VOICES = ["Aoede", "Kore", "Zephyr", "Puck", "Charon", "Fenrir"]


def get_speaker_index():
    """Finds output device index matching G-MARK Micro Go USB."""
    devices = sd.query_devices()
    for idx, d in enumerate(devices):
        if AUDIO_DEVICE_NAME.lower() in d.get("name", "").lower():
            if d.get("max_output_channels", 0) > 0:
                return idx
    return sd.default.device[1]


async def audition_voice(voice_name: str, phrase: str, play: bool = True, save_path: Path = None):
    print("=" * 65)
    print(f"AUDITIONING GEMINI LIVE VOICE: '{voice_name}'")
    print(f"Model: {GEMINI_LIVE_MODEL}")
    print(f"Phrase: \"{phrase}\"")
    print("=" * 65)

    client = genai.Client(api_key=GOOGLE_API_KEY)
    config = types.LiveConnectConfig(
        response_modalities=["AUDIO"],
        speech_config=types.SpeechConfig(
            voice_config=types.VoiceConfig(
                prebuilt_voice_config=types.PrebuiltVoiceConfig(voice_name=voice_name)
            )
        )
    )

    print("Connecting to Gemini Live...")
    pcm_chunks = bytearray()
    sample_rate = 24000

    async with client.aio.live.connect(model=GEMINI_LIVE_MODEL, config=config) as session:
        prompt_text = f"Speak the following sentence clearly and naturally: {phrase}"
        await session.send_client_content(
            turns=[types.Content(role="user", parts=[types.Part.from_text(text=prompt_text)])],
            turn_complete=True
        )

        async for response in session.receive():
            sc = response.server_content
            if sc and sc.model_turn:
                for part in sc.model_turn.parts:
                    if part.inline_data and part.inline_data.data:
                        pcm_chunks.extend(part.inline_data.data)
                        if "rate=" in (part.inline_data.mime_type or ""):
                            try:
                                sample_rate = int(part.inline_data.mime_type.split("rate=")[-1].split(";")[0])
                            except ValueError:
                                pass
            if sc and sc.turn_complete:
                break

    if not pcm_chunks:
        print("❌ No audio data received from Gemini Live.")
        return

    # Convert PCM16 to float32
    audio_data = np.frombuffer(pcm_chunks, dtype=np.int16).astype(np.float32) / 32768.0
    duration = len(audio_data) / sample_rate
    peak = float(np.max(np.abs(audio_data))) if len(audio_data) > 0 else 0.0
    rms = float(np.sqrt(np.mean(audio_data ** 2))) if len(audio_data) > 0 else 0.0

    print(f"✅ Received {duration:.2f}s of audio ({len(pcm_chunks)} bytes at {sample_rate} Hz)")
    print(f"   Peak amplitude: {peak:.3f} | RMS: {rms:.3f}")

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(save_path), audio_data, sample_rate)
        print(f"💾 Saved audition recording to: {save_path}")

    if play:
        out_idx = get_speaker_index()
        dev_info = sd.query_devices(out_idx)
        print(f"🔊 Playing audio on output [{out_idx}] ({dev_info.get('name', 'Default')})...")

        # Upsample 24k -> 48k stereo if playing to G-MARK Micro Go (native 48k device)
        if dev_info.get("max_output_channels", 0) >= 2:
            audio_48k = np.repeat(audio_data, 2)
            stereo_48k = np.column_stack((audio_48k, audio_48k)) * 0.5  # Safe gain
            sd.play(stereo_48k, samplerate=48000, device=out_idx)
        else:
            sd.play(audio_data, samplerate=sample_rate, device=out_idx)
        sd.wait()
        print("✅ Playback finished.")


def main():
    parser = argparse.ArgumentParser(description="Audition Gemini Live voices on Raspberry Pi 5")
    parser.add_argument(
        "--voice",
        choices=SUPPORTED_VOICES,
        default="Kore",
        help="Voice ID to audition (default: Kore)"
    )
    parser.add_argument(
        "--phrase",
        default="Hello! I am Berry Good Listener, your voice assistant on Raspberry Pi 5. How does my voice sound to you?",
        help="Custom phrase for the voice to say"
    )
    parser.add_argument(
        "--no-play",
        action="store_true",
        help="Do not play audio out of speaker (save only or benchmark)"
    )
    parser.add_argument(
        "--save",
        type=Path,
        default=None,
        help="Optional path to save audition WAV file"
    )
    args = parser.parse_args()

    save_target = args.save
    if save_target is None:
        save_target = BASE_DIR / "tests" / "recordings" / f"audition_{args.voice.lower()}.wav"

    asyncio.run(audition_voice(args.voice, args.phrase, play=not args.no_play, save_path=save_target))


if __name__ == "__main__":
    main()

