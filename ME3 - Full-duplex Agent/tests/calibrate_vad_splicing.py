#!/usr/bin/env python3
"""
ME3 VAD & Audio Splicing Calibration Utility
Diagnoses why commands like 'Turn on the light' get misrecognized or sliced into 'I am' / 'Bye' / 'Get up'.

Features:
1. Live Microphone Capture with Real-Time VAD State Machine:
   - Displays real-time VU meter and Silero VAD state (QUIET, STARTING, SPEAKING, STOPPING).
2. Live Turn Splicing & Inspection:
   - When you speak, captures the EXACT sliced audio segment that VAD creates.
   - Saves each slice to `tests/recordings/live_slice_<index>.wav`.
   - Plays the captured slice back to you through the speaker so you can hear if your words were cut off.
3. End-to-End Gemini Live Verification:
   - Streams the slice into Gemini Live API.
   - Displays Google's cloud transcription [Transcription:user] and dispatched tool call.
4. Mode Comparison:
   - Option [1]: Client-Driven VAD (Tests local Silero slicing & silence threshold).
   - Option [2]: Server-Driven VAD (Tests continuous streaming with echo gate).
   - Option [3]: Interactive VAD Parameter Tuning (test different silence_ms and confidence).
"""

import sys
import time
import queue
import asyncio
import threading
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import scipy.signal
import sounddevice as sd
import soundfile as sf

from config import (
    GOOGLE_API_KEY, GEMINI_LIVE_MODEL, GEMINI_VOICE_ID, GEMINI_LANGUAGE, SYSTEM_INSTRUCTION,
    AUDIO_HARDWARE_RATE, AUDIO_MODEL_RATE, VAD_CONFIDENCE_THRESHOLD, VAD_STOP_SPEECH_MS
)
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.audio.vad.vad_analyzer import VADParams, VADState
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, GeminiModalities, GeminiVADParams
from pipecat.frames.frames import (
    Frame, InputAudioRawFrame, TTSAudioRawFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame,
    FunctionCallInProgressFrame, TextFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.aggregators.llm_response_universal import LLMContext, LLMContextAggregatorPair
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from agent.pipeline import Gemini3LiveService
from mcp_server import create_mcp_server
from mcp_server.client_bridge import get_mcp_function_schemas
from hardware import HardwareManager


def get_audio_devices():
    devices = sd.query_devices()
    in_idx, out_idx = None, None
    for idx, dev in enumerate(devices):
        name = dev["name"].lower()
        if "usb composite device" in name or "g-mark" in name or "micro go" in name:
            if dev["max_input_channels"] > 0 and in_idx is None:
                in_idx = idx
            if dev["max_output_channels"] > 0 and out_idx is None:
                out_idx = idx
    if in_idx is None:
        in_idx = sd.default.device[0]
    if out_idx is None:
        out_idx = sd.default.device[1]
    return in_idx, out_idx


async def test_slice_against_gemini(pcm_16k_bytes: bytes, tool_schemas: list, mcp_serv=None, mock_hw=None, vad_mode: str = "client"):
    """
    Feeds a sliced PCM audio segment to Gemini Live and prints the transcription and tool calls.
    """
    from mcp_server.client_bridge import bridge_mcp_to_pipecat

    settings = GeminiLiveLLMService.Settings(
        model=GEMINI_LIVE_MODEL,
        voice=GEMINI_VOICE_ID,
        modalities=GeminiModalities.AUDIO,
        language=GEMINI_LANGUAGE,
        system_instruction=SYSTEM_INSTRUCTION,
        vad=GeminiVADParams(disabled=True) if vad_mode == "client" else None
    )
    llm = Gemini3LiveService(api_key=GOOGLE_API_KEY, settings=settings, tools=tool_schemas, hw_mgr=mock_hw)
    if mcp_serv:
        bridge_mcp_to_pipecat(mcp_serv, llm, None, mock_hw)

    transcriptions = []
    tool_calls = []
    spoken_replies = []
    response_event = asyncio.Event()

    class Monitor(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, FunctionCallInProgressFrame):
                tool_desc = f"{frame.function_name}({frame.arguments})"
                tool_calls.append(tool_desc)
                response_event.set()
            elif isinstance(frame, TTSAudioRawFrame):
                spoken_replies.append(frame.audio)
                response_event.set()
            elif isinstance(frame, TextFrame):
                if frame.text.strip():
                    transcriptions.append(frame.text)
            await self.push_frame(frame, direction)

    context = LLMContext()
    user_agg, asst_agg = LLMContextAggregatorPair(context)
    llm._context = context

    monitor = Monitor()
    pipeline = Pipeline([user_agg, llm, monitor, asst_agg])
    runner = PipelineRunner()
    task = PipelineTask(pipeline)
    runner_task = asyncio.create_task(runner.run(task))

    for _ in range(50):
        if llm._session:
            break
        await asyncio.sleep(0.1)

    if not llm._session:
        print("❌ Could not connect to Gemini Live.")
        await task.cancel()
        return transcriptions, tool_calls, spoken_replies

    llm._ready_for_realtime_input = True
    await task.queue_frame(UserStartedSpeakingFrame())

    chunk_size = 1024
    for i in range(0, len(pcm_16k_bytes), chunk_size):
        chunk = pcm_16k_bytes[i:i+chunk_size]
        await task.queue_frame(InputAudioRawFrame(audio=chunk, sample_rate=16000, num_channels=1))
        await asyncio.sleep(0.015)

    await task.queue_frame(UserStoppedSpeakingFrame())

    try:
        await asyncio.wait_for(response_event.wait(), timeout=6.0)
        await asyncio.sleep(0.8)
    except asyncio.TimeoutError:
        pass

    await task.cancel()
    try:
        await runner_task
    except Exception:
        pass

    return transcriptions, tool_calls, spoken_replies


def run_splicing_calibrator(confidence_threshold=0.45, stop_speech_ms=650, vad_mode="client"):
    in_idx, out_idx = get_audio_devices()
    print("=" * 70)
    print("      ME3 LIVE VAD & SPLICING CALIBRATION & DIAGNOSTIC TOOL       ")
    print("=" * 70)
    print(f"Microphone: Device [{in_idx}] | Speaker: Device [{out_idx}]")
    print(f"VAD Threshold: {confidence_threshold:.2f} | Turn Stop Silence: {stop_speech_ms} ms | Gemini VAD Mode: {vad_mode}")
    print("-" * 70)
    print("This tool lets you speak naturally (e.g. 'Turn on the lights') and:")
    print(" 1. Measures exactly how local VAD slices your words.")
    print(" 2. Saves the slice to disk.")
    print(" 3. Plays back the slice so you can hear if words were cut.")
    print(" 4. Sends the slice to Gemini Live to see what it heard and executed.")
    print("-" * 70)

    # Initialize Hardware schemas
    mock_hw = HardwareManager(mock=True)
    mcp_serv = create_mcp_server(mock_hw)
    tool_schemas = get_mcp_function_schemas(mcp_serv)

    # Prepare VAD
    vad_analyzer = SileroVADAnalyzer(
        params=VADParams(
            confidence=confidence_threshold,
            start_secs=0.10,
            stop_secs=stop_speech_ms / 1000.0
        )
    )
    vad_analyzer.set_sample_rate(16000)

    # DSP Filters
    sos_lpf = scipy.signal.butter(4, 7500, btype="lowpass", fs=48000, output="sos")
    lpf_zi = scipy.signal.sosfilt_zi(sos_lpf)
    sos_hpf = scipy.signal.butter(2, 100, btype="highpass", fs=16000, output="sos")
    hpf_zi = scipy.signal.sosfilt_zi(sos_hpf)

    audio_queue = queue.Queue()
    running = True

    def input_callback(indata, frames, time_info, status):
        nonlocal lpf_zi, hpf_zi
        if not running:
            return
        samples_48k = indata[:, 0]
        filt_48k, lpf_zi = scipy.signal.sosfilt(sos_lpf, samples_48k, zi=lpf_zi)
        samples_16k = filt_48k[::3].astype(np.float32)
        filtered_16k, hpf_zi = scipy.signal.sosfilt(sos_hpf, samples_16k, zi=hpf_zi)
        boosted = np.clip(filtered_16k * 1.15, -0.95, 0.95)
        audio_queue.put(boosted)

    stream = sd.InputStream(
        device=in_idx,
        channels=1,
        samplerate=48000,
        blocksize=1536,
        dtype="float32",
        callback=input_callback
    )
    stream.start()

    out_dir = BASE_DIR / "tests" / "recordings"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n🟢 LISTENING LIVE... Speak a command now (e.g. 'Turn on the lights' or 'Tell me a joke').")
    print("Press Ctrl+C to stop.\n")

    user_speaking = False
    speech_start_ts = 0.0
    last_speech_ts = 0.0
    onset_streak = 0
    slice_buffer = []
    turn_count = 0

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    try:
        while True:
            try:
                chunk_16k = audio_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            rms = float(np.sqrt(np.mean(chunk_16k ** 2)))
            pcm_bytes = (chunk_16k * 32767).astype(np.int16).tobytes()

            conf = float(vad_analyzer.voice_confidence(pcm_bytes)[0]) if len(pcm_bytes) >= 512 else 0.0
            vad_state = loop.run_until_complete(vad_analyzer.analyze_audio(pcm_bytes))

            is_speech = (conf >= confidence_threshold or vad_state == VADState.SPEAKING) and rms >= 0.015
            now = time.perf_counter()

            # Visual meter
            bars = int(min(1.0, rms / 0.35) * 15)
            meter = "█" * bars + "░" * (15 - bars)
            state_str = vad_state.name.ljust(8)

            if is_speech:
                onset_streak += 1
            else:
                onset_streak = 0

            # Turn onset
            if not user_speaking:
                # Keep rolling pre-roll of ~300ms (10 chunks)
                slice_buffer.append(chunk_16k)
                if len(slice_buffer) > 10:
                    slice_buffer.pop(0)

                sys.stdout.write(f"\r[IDLE    ] [{meter}] RMS: {rms:.3f} | Conf: {conf:.2f} | State: {state_str} ")
                sys.stdout.flush()

                if onset_streak >= 2 or (vad_state == VADState.STARTING and is_speech):
                    user_speaking = True
                    speech_start_ts = now - 0.064
                    last_speech_ts = now
                    onset_streak = 0
                    print(f"\n🎙️  [TURN START DETECTED] at t={time.strftime('%H:%M:%S')}")
            else:
                slice_buffer.append(chunk_16k)
                if is_speech or (vad_state == VADState.STOPPING and rms >= 0.015):
                    last_speech_ts = now

                silence_dur = now - last_speech_ts
                voice_dur = last_speech_ts - speech_start_ts

                sys.stdout.write(f"\r[SPEAKING] [{meter}] Voice: {voice_dur:.2f}s | Silence: {silence_dur:.2f}s | Conf: {conf:.2f} | State: {state_str} ")
                sys.stdout.flush()

                # Turn end condition
                silence_threshold = stop_speech_ms / 1000.0
                is_turn_end = (vad_state == VADState.QUIET and silence_dur >= 0.45) or (silence_dur >= silence_threshold)

                if is_turn_end:
                    user_speaking = False
                    onset_streak = 0
                    total_dur = len(slice_buffer) * 0.032

                    if voice_dur >= 0.20:
                        turn_count += 1
                        print(f"\n🛑 [TURN END DETECTED] Total audio captured: {total_dur:.2f}s (voice: {voice_dur:.2f}s, silence: {silence_dur:.2f}s)")
                        
                        full_slice = np.concatenate(slice_buffer)
                        slice_buffer = []

                        slice_filename = f"live_splice_turn_{turn_count}.wav"
                        slice_path = out_dir / slice_filename
                        sf.write(str(slice_path), full_slice, 16000)
                        print(f"💾 Saved captured slice to: tests/recordings/{slice_filename}")

                        # 1. Play back slice to user
                        print("🔊 Playing back captured audio slice through speaker...")
                        play_48k = np.repeat(full_slice, 3)  # 16k -> 48k
                        stereo_48k = np.column_stack((play_48k, play_48k))
                        try:
                            sd.play(stereo_48k, samplerate=48000, device=out_idx)
                            sd.wait()
                        except Exception as e:
                            print(f"Playback error: {e}")

                        # 2. Test slice against Gemini Live
                        print(f"🌐 Testing slice against Gemini Live API (mode: {vad_mode})...")
                        pcm_slice_bytes = (full_slice * 32767).astype(np.int16).tobytes()
                        texts, tools, replies = loop.run_until_complete(
                            test_slice_against_gemini(
                                pcm_slice_bytes, tool_schemas,
                                mcp_serv=mcp_serv, mock_hw=mock_hw, vad_mode=vad_mode
                            )
                        )

                        print("-" * 50)
                        if tools:
                            print(f"🎯 [GEMINI TOOL CALL]: {tools[0]}")
                        if texts:
                            print(f"💬 [GEMINI TEXT]: {' '.join(texts)}")
                        if not tools and not texts:
                            print("⚠️ [GEMINI]: No tool call or text generated.")
                        print("-" * 50)
                        print("\n🟢 READY FOR NEXT PHRASE. Speak again or Ctrl+C to finish.\n")
                    else:
                        print("\n⚠️ Ignored transient noise (<0.20s voice).")
                        slice_buffer = []

    except KeyboardInterrupt:
        print("\n\nExiting calibration...")
    finally:
        running = False
        stream.stop()
        stream.close()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ME3 Live VAD & Splicing Calibration Utility")
    parser.add_argument("--threshold", type=float, default=0.45, help="Silero VAD confidence threshold (default: 0.45)")
    parser.add_argument("--stop-ms", type=int, default=650, help="VAD silence duration in ms to end turn (default: 650)")
    parser.add_argument("--vad-mode", choices=["client", "server"], default="client", help="Gemini Live VAD mode: client or server (default: client)")
    args = parser.parse_args()

    run_splicing_calibrator(confidence_threshold=args.threshold, stop_speech_ms=args.stop_ms, vad_mode=args.vad_mode)
