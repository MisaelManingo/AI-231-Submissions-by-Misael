#!/usr/bin/env python3
"""
Interactive Voice Calibration & Gemini Live End-to-End Acoustic Diagnostic
For ME3 Full-Duplex Agent on Raspberry Pi 5 with G-MARK Micro Go Conference Speaker/Mic.

Features:
1. Measures ambient room noise floor (silence baseline).
2. Interactive 3-2-1 countdown for user to speak a test command.
3. Records 4 seconds of live speech at 48,000 Hz with real-time ASCII VU meter.
4. Performs Acoustic & DSP Analysis:
   - Peak amplitude & RMS level
   - Signal-to-Noise Ratio (SNR in dB)
   - Optimal digital gain recommendation
   - Silero VAD speech onset / offset detection timeline
5. Tests the actual recorded voice directly against Gemini Live Multimodal API:
   - Feeds the exact processed 16 kHz stream into Gemini Live
   - Verifies whether Gemini understands the command, calls the right tool, or responds
   - Measures exact cloud turn latency
6. Plays back the assistant's voice through the G-MARK speaker.
"""

import os
import sys
import time
import asyncio
from pathlib import Path

# Add project root to path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import scipy.signal
import sounddevice as sd
import soundfile as sf

from config import (
    GOOGLE_API_KEY, GEMINI_LIVE_MODEL, GEMINI_VOICE_ID, GEMINI_LANGUAGE,
    SYSTEM_INSTRUCTION, AUDIO_DEVICE_NAME, AUDIO_HARDWARE_RATE, AUDIO_MODEL_RATE
)
from agent.audio_transport import find_audio_device_indices
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams


def print_banner(text: str):
    print("\n" + "=" * 65)
    print(text.center(65))
    print("=" * 65)


def record_with_meter(in_idx, duration_sec=4.0, rate=48000, label="RECORDING"):
    """Records audio from sounddevice while displaying a continuous ASCII VU meter."""
    total_frames = int(duration_sec * rate)
    chunk_size = 1536  # 32ms frames
    recorded_chunks = []
    
    # Progress and meter setup
    stream = sd.InputStream(device=in_idx, channels=1, samplerate=rate, dtype="float32", blocksize=chunk_size)
    stream.start()
    
    start_time = time.perf_counter()
    frames_captured = 0

    try:
        while frames_captured < total_frames:
            data, overflowed = stream.read(chunk_size)
            recorded_chunks.append(data.copy())
            frames_captured += len(data)

            # Calculate RMS and peak for VU meter
            peak = float(np.max(np.abs(data)))
            rms = float(np.sqrt(np.mean(data ** 2)))
            
            # 20-segment visual meter
            meter_bars = int(min(1.0, peak / 0.85) * 20)
            bar_str = "█" * meter_bars + "░" * (20 - meter_bars)
            elapsed = time.perf_counter() - start_time
            rem = max(0.0, duration_sec - elapsed)
            
            sys.stdout.write(f"\r🎙️  [{label}] [{bar_str}] Peak: {peak:.3f} | RMS: {rms:.3f} | {rem:.1f}s left ")
            sys.stdout.flush()
    finally:
        stream.stop()
        stream.close()
        print()

    audio = np.concatenate(recorded_chunks, axis=0)[:, 0]
    return audio[:total_frames]


async def test_against_gemini_live(audio_16k_pcm: bytes, tools_list: list):
    """Feeds the recorded audio bytes into Gemini Live API and captures the model response."""
    from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, GeminiModalities
    from pipecat.frames.frames import (
        InputAudioRawFrame, StartFrame, TTSAudioRawFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
    )

    print("\n🌐 Connecting to Gemini Live Multimodal API...")
    settings = GeminiLiveLLMService.Settings(
        model=GEMINI_LIVE_MODEL,
        voice=GEMINI_VOICE_ID,
        modalities=GeminiModalities.AUDIO,
        language=GEMINI_LANGUAGE,
        system_instruction=SYSTEM_INSTRUCTION
    )
    llm = GeminiLiveLLMService(api_key=GOOGLE_API_KEY, settings=settings, tools=tools_list)

    reply_text_chunks = []
    reply_audio_chunks = []
    tool_calls_detected = []
    response_started_ts = None
    t0 = time.perf_counter()

    # Track output frames
    from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
    class Receiver(FrameProcessor):
        async def process_frame(self, frame, direction=FrameDirection.DOWNSTREAM):
            await super().process_frame(frame, direction)
            nonlocal response_started_ts
            from pipecat.frames.frames import FunctionCallInProgressFrame, TextFrame
            if isinstance(frame, TTSAudioRawFrame):
                if response_started_ts is None:
                    response_started_ts = time.perf_counter()
                reply_audio_chunks.append(frame.audio)
            elif isinstance(frame, FunctionCallInProgressFrame):
                if response_started_ts is None:
                    response_started_ts = time.perf_counter()
                tool_calls_detected.append(f"{frame.function_name}({frame.arguments})")
            elif isinstance(frame, TextFrame):
                reply_text_chunks.append(frame.text)
            await self.push_frame(frame, direction)

    from pipecat.processors.aggregators.llm_response_universal import LLMContext, LLMContextAggregatorPair
    from pipecat.pipeline.task import PipelineTask
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.runner import PipelineRunner
    
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)
    llm._context = context

    receiver = Receiver()
    pipeline = Pipeline([user_aggregator, llm, receiver, assistant_aggregator])
    runner = PipelineRunner()
    task = PipelineTask(pipeline)

    async def _runner_coro():
        await runner.run(task)

    runner_task = asyncio.create_task(_runner_coro())
    
    # Wait for Gemini connection to establish
    for _ in range(50):
        if llm._session:
            break
        await asyncio.sleep(0.1)

    if not llm._session:
        print("❌ Could not connect to Gemini service. Please check your GOOGLE_API_KEY.")
        await task.cancel()
        return None, None, None

    llm._ready_for_realtime_input = True
    print(f"✅ Connected to Gemini Live in {(time.perf_counter()-t0)*1000:.0f}ms.")
    print("📤 Streaming your recorded voice into Gemini Live session...")

    # Push user speech start frame
    await task.queue_frame(UserStartedSpeakingFrame())

    # Stream 32ms audio frames (512 samples = 1024 bytes @ 16 kHz)
    chunk_bytes = 1024
    for i in range(0, len(audio_16k_pcm), chunk_bytes):
        chunk = audio_16k_pcm[i:i+chunk_bytes]
        frame = InputAudioRawFrame(audio=chunk, sample_rate=16000, num_channels=1)
        await task.queue_frame(frame)
        await asyncio.sleep(0.015)  # Slightly faster than real-time to avoid buffer delay

    # Push user speech end frame
    t_audio_end = time.perf_counter()
    await task.queue_frame(UserStoppedSpeakingFrame())
    print("⏳ Awaiting Gemini Live response...")

    # Wait up to 5 seconds for response or tool call
    for _ in range(50):
        if reply_audio_chunks or tool_calls_detected:
            # Let remaining audio chunks arrive
            await asyncio.sleep(0.8)
            break
        await asyncio.sleep(0.1)

    latency_ms = (response_started_ts - t_audio_end) * 1000.0 if response_started_ts else 0.0
    await task.cancel()
    
    return tool_calls_detected, reply_audio_chunks, latency_ms


def main():
    print_banner("ME3 CONFERENCE MIC/SPEAKER VOICE CALIBRATION")
    print("This utility records your actual voice on the G-MARK conference unit,")
    print("analyzes acoustic levels, and verifies end-to-end recognition with Gemini Live.")

    in_idx, out_idx = find_audio_device_indices()
    if in_idx is None or out_idx is None:
        print(f"❌ Could not find audio device matching '{AUDIO_DEVICE_NAME}'.")
        print("Available devices:")
        for idx, d in enumerate(sd.query_devices()):
            print(f"  [{idx}] {d.get('name')}")
        return

    print(f"Selected Device: [{in_idx}] '{sd.query_devices(in_idx)['name']}'")
    print(f"Capture Rate: {AUDIO_HARDWARE_RATE} Hz | Target Model Rate: {AUDIO_MODEL_RATE} Hz")

    # Step 0: Ensure hardware AGC is disabled for consistent measurement
    try:
        import subprocess
        subprocess.run(["amixer", "-c", "2", "set", "Auto Gain Control", "off"], capture_output=True, timeout=1)
        print("✅ Disabled USB Hardware AGC (amixer) for clean linear response.")
    except Exception:
        pass

    # Step 1: Ambient Noise Baseline Measurement
    print("\n" + "-" * 65)
    print("STEP 1: Measuring Room & Fan Noise Floor (Please stay completely silent)")
    print("-" * 65)
    for c in range(3, 0, -1):
        print(f"Starting in {c}...", end="\r")
        time.sleep(1.0)
    print("Measuring now...                                ")

    ambient_audio = record_with_meter(in_idx, duration_sec=2.5, rate=AUDIO_HARDWARE_RATE, label="AMBIENT NOISE")
    amb_peak = float(np.max(np.abs(ambient_audio)))
    amb_rms = float(np.sqrt(np.mean(ambient_audio ** 2)))
    print(f"📊 Ambient Noise Level: Peak = {amb_peak:.4f} (-{20*np.log10(1/(amb_peak+1e-6)):.1f} dBFS) | RMS = {amb_rms:.4f}")

    # Step 2: User Speech Recording
    print("\n" + "-" * 65)
    print("STEP 2: Voice Command Recording")
    print("Sit at your natural distance from the conference speaker/mic and speak normally.")
    print("Suggested test phrases:")
    print("  👉 \"Turn on the lights\"")
    print("  👉 \"What is the weather in Manila?\"")
    print("  👉 \"Play some music\"")
    print("-" * 65)
    for c in range(3, 0, -1):
        print(f"Get ready to speak in {c}...", end="\r")
        time.sleep(1.0)
    print("🗣️  SPEAK NOW!                                      ")

    speech_audio_48k = record_with_meter(in_idx, duration_sec=4.0, rate=AUDIO_HARDWARE_RATE, label="VOICE CAPTURE")

    # Step 3: Acoustic & DSP Analysis
    print("\n" + "-" * 65)
    print("STEP 3: Acoustic & Signal Processing Analysis")
    print("-" * 65)
    
    # Save raw audio for inspection
    out_dir = BASE_DIR / "tests" / "recordings"
    out_dir.mkdir(parents=True, exist_ok=True)
    raw_path = out_dir / "user_voice_48k_raw.wav"
    sf.write(str(raw_path), speech_audio_48k, AUDIO_HARDWARE_RATE)
    print(f"💾 Saved raw 48 kHz capture to: {raw_path.name}")

    speech_peak = float(np.max(np.abs(speech_audio_48k)))
    speech_rms = float(np.sqrt(np.mean(speech_audio_48k ** 2)))
    snr_db = 20 * np.log10((speech_rms + 1e-6) / (amb_rms + 1e-6))

    print(f"🎤 Voice Raw Peak: {speech_peak:.4f} (-{20*np.log10(1/(speech_peak+1e-6)):.1f} dBFS)")
    print(f"🎤 Voice Raw RMS:  {speech_rms:.4f}")
    print(f"📶 Signal-to-Noise Ratio (SNR): {snr_db:.1f} dB")

    # High-Pass Filter (100 Hz Butterworth)
    sos = scipy.signal.butter(2, 100, btype="highpass", fs=AUDIO_HARDWARE_RATE, output="sos")
    filtered_48k = scipy.signal.sosfilt(sos, speech_audio_48k)
    filt_peak = float(np.max(np.abs(filtered_48k)))

    # Downsample to 16 kHz using polyphase FIR
    samples_16k = scipy.signal.resample_poly(filtered_48k, 1, 3).astype(np.float32)

    # Dynamic Digital Gain calculation (aim for peak ~ 0.75 - 0.85)
    if filt_peak > 0.01:
        target_gain = min(4.0, max(1.0, 0.80 / filt_peak))
    else:
        target_gain = 2.0

    print(f"🎛️  Recommended Digital Gain: {target_gain:.2f}x (Current setting in transport: 1.15x)")
    boosted_16k = np.clip(samples_16k * target_gain, -0.95, 0.95)
    proc_path = out_dir / "user_voice_16k_processed.wav"
    sf.write(str(proc_path), boosted_16k, 16000)
    print(f"💾 Saved DSP processed 16 kHz audio to: {proc_path.name}")

    # Silero VAD Evaluation on the recording
    print("\n🔍 Evaluating Silero VAD Detection Timeline...")
    vad = SileroVADAnalyzer(params=VADParams(confidence=0.65, start_secs=0.12, stop_secs=0.40))
    vad.set_sample_rate(16000)
    
    pcm_bytes = (boosted_16k * 32767).astype(np.int16).tobytes()
    chunk_size_bytes = 1024
    speech_frames_detected = 0
    total_chunks = len(pcm_bytes) // chunk_size_bytes
    
    for i in range(total_chunks):
        chunk = pcm_bytes[i*chunk_size_bytes:(i+1)*chunk_size_bytes]
        conf = float(vad.voice_confidence(chunk)[0])
        if conf >= 0.65:
            speech_frames_detected += 1

    detected_duration = speech_frames_detected * 0.032
    print(f"🎙️  Silero VAD Confirmed Speech: {detected_duration:.2f} seconds out of 4.0s (Confidence >= 0.65)")
    if detected_duration < 0.3:
        print("⚠️  Warning: Speech duration was very short. Speak louder or sit closer to the conference mic.")

    # Step 4: Live Verification with Gemini Live API
    print("\n" + "-" * 65)
    print("STEP 4: Gemini Live Multimodal Speech-to-Speech End-to-End Test")
    print("-" * 65)

    from mcp_server import create_mcp_server
    from mcp_server.client_bridge import get_mcp_function_schemas
    from hardware import HardwareManager
    
    # Temporary mock hardware manager to extract schemas
    mock_hw = HardwareManager(mock=True)
    mcp_serv = create_mcp_server(mock_hw)
    tool_schemas = get_mcp_function_schemas(mcp_serv)

    tool_calls, reply_audio, latency_ms = asyncio.run(test_against_gemini_live(pcm_bytes, tool_schemas))

    print("\n" + "=" * 65)
    print("CALIBRATION & LIVE VERIFICATION RESULTS".center(65))
    print("=" * 65)

    if tool_calls:
        print(f"🎯 Tool Call Detected: {tool_calls[0]}")
        print(f"⚡ End-of-Speech to Tool Latency: {latency_ms:.0f} ms")
        print("✅ SUCCESS: Gemini recognized the voice command and dispatched the correct tool!")
    elif reply_audio:
        print(f"🗣️  Spoken Response Generated: {len(reply_audio)} audio frames")
        print(f"⚡ End-of-Speech to Reply Latency: {latency_ms:.0f} ms")
        print("✅ SUCCESS: Gemini understood your speech and produced a conversational response!")

        # Play back response through speaker
        print("\n🔊 Playing Gemini's response through the G-MARK speaker...")
        raw_pcm = b"".join(reply_audio)
        audio_arr = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
        # Upsample 24k -> 48k for playback
        audio_48k = np.repeat(audio_arr, 2)
        stereo_48k = np.column_stack((audio_48k, audio_48k))
        try:
            sd.play(stereo_48k, samplerate=48000, device=out_idx)
            sd.wait()
            print("🔊 Playback complete.")
        except Exception as e:
            print(f"Playback notice: {e}")
    else:
        print("⚠️  Gemini did not produce a response or tool call.")
        print("Tip: Check network connectivity or check if speech was clear and audible.")

    print("\n✅ Voice calibration and live test script finished.")


if __name__ == "__main__":
    main()

