#!/usr/bin/env python3
"""
Test the user's real recorded voice against Gemini Live API.
Uses the 48 kHz raw recording saved during calibration:
1. Filters out sub-100 Hz rumble
2. Decimates 3:1 to 16 kHz using polyphase FIR
3. Applies calibrated 1.15x gain
4. Streams audio frames into Gemini Live session
5. Captures Gemini's recognized command, tool call, and spoken response!
"""

import sys
import asyncio
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import scipy.signal
import soundfile as sf

from config import (
    GOOGLE_API_KEY, GEMINI_LIVE_MODEL, GEMINI_VOICE_ID, GEMINI_LANGUAGE, SYSTEM_INSTRUCTION
)
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, GeminiModalities
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.frames.frames import (
    Frame, InputAudioRawFrame, TTSAudioRawFrame, UserStartedSpeakingFrame, UserStoppedSpeakingFrame
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from mcp_server import create_mcp_server
from mcp_server.client_bridge import get_mcp_function_schemas
from hardware import HardwareManager


async def main():
    wav_path = BASE_DIR / "tests" / "recordings" / "user_voice_48k_raw.wav"
    if not wav_path.exists():
        print(f"File not found: {wav_path}")
        return

    print("=" * 60)
    print("TESTING RECORDED USER VOICE WITH GEMINI LIVE")
    print("=" * 60)
    
    # 1. Load and process audio
    raw_48k, sr = sf.read(str(wav_path), dtype="float32")
    print(f"Loaded {len(raw_48k)/sr:.2f}s of raw audio (peak={np.max(np.abs(raw_48k)):.3f}, rms={np.sqrt(np.mean(raw_48k**2)):.3f})")

    # 100 Hz HPF
    sos = scipy.signal.butter(2, 100, btype="highpass", fs=48000, output="sos")
    filt_48k = scipy.signal.sosfilt(sos, raw_48k)

    # 3:1 Polyphase downsampling
    samples_16k = scipy.signal.resample_poly(filt_48k, 1, 3).astype(np.float32)

    # Calibrated gain: 1.15x
    calibrated_gain = 1.15
    boosted_16k = np.clip(samples_16k * calibrated_gain, -0.95, 0.95)
    print(f"Processed 16 kHz audio: peak={np.max(np.abs(boosted_16k)):.3f}, rms={np.sqrt(np.mean(boosted_16k**2)):.3f}")

    pcm_bytes = (boosted_16k * 32767).astype(np.int16).tobytes()

    # 2. Setup Gemini Live with tools
    mock_hw = HardwareManager(mock=True)
    mcp_serv = create_mcp_server(mock_hw)
    tool_schemas = get_mcp_function_schemas(mcp_serv)

    settings = GeminiLiveLLMService.Settings(
        model=GEMINI_LIVE_MODEL,
        voice=GEMINI_VOICE_ID,
        modalities=GeminiModalities.AUDIO,
        language=GEMINI_LANGUAGE,
        system_instruction=SYSTEM_INSTRUCTION
    )
    llm = GeminiLiveLLMService(api_key=GOOGLE_API_KEY, settings=settings, tools=tool_schemas)

    tool_calls = []
    audio_replies = []
    transcriptions = []
    response_event = asyncio.Event()

    class OutputMonitor(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            from pipecat.frames.frames import FunctionCallInProgressFrame, TextFrame, TranscriptionFrame
            if isinstance(frame, FunctionCallInProgressFrame):
                call_desc = f"{frame.function_name}({frame.arguments})"
                print(f"🎯 [GEMINI TOOL CALL]: {call_desc}")
                tool_calls.append(call_desc)
                response_event.set()
            elif isinstance(frame, TTSAudioRawFrame):
                audio_replies.append(frame.audio)
                response_event.set()
            elif isinstance(frame, (TextFrame, TranscriptionFrame)):
                text = getattr(frame, "text", "")
                if text:
                    print(f"💬 [GEMINI TEXT]: {text}")
                    transcriptions.append(text)
            await self.push_frame(frame, direction)

    from pipecat.processors.aggregators.llm_response_universal import LLMContext, LLMContextAggregatorPair
    context = LLMContext()
    user_aggregator, assistant_aggregator = LLMContextAggregatorPair(context)
    llm._context = context

    monitor = OutputMonitor()
    pipeline = Pipeline([user_aggregator, llm, monitor, assistant_aggregator])
    runner = PipelineRunner()
    task = PipelineTask(pipeline)

    runner_task = asyncio.create_task(runner.run(task))

    print("Connecting to Gemini Live WebSocket...")
    for _ in range(50):
        if llm._session:
            break
        await asyncio.sleep(0.1)

    if not llm._session:
        print("❌ Could not connect to Gemini service.")
        await task.cancel()
        return

    # Ensure ready for realtime input
    llm._ready_for_realtime_input = True
    print("Connected! Streaming user voice frames...")
    await task.queue_frame(UserStartedSpeakingFrame())

    chunk_bytes = 1024  # 32ms @ 16 kHz
    for i in range(0, len(pcm_bytes), chunk_bytes):
        chunk = pcm_bytes[i:i+chunk_bytes]
        frame = InputAudioRawFrame(audio=chunk, sample_rate=16000, num_channels=1)
        await task.queue_frame(frame)
        await asyncio.sleep(0.02)  # Stream in near real-time

    print("Stream finished. Emitting speech end...")
    await task.queue_frame(UserStoppedSpeakingFrame())

    print("Waiting for Gemini response...")
    try:
        await asyncio.wait_for(response_event.wait(), timeout=7.0)
        # Give remaining audio chunks 1 second to arrive
        await asyncio.sleep(1.0)
    except asyncio.TimeoutError:
        print("⏱️ Timeout waiting for Gemini response.")

    print("\n" + "=" * 60)
    print("VERIFICATION RESULTS:")
    print("=" * 60)
    if tool_calls:
        print(f"✅ Success! Gemini executed Tool: {tool_calls[0]}")
    if audio_replies:
        total_audio_bytes = sum(len(c) for c in audio_replies)
        print(f"✅ Success! Gemini generated {len(audio_replies)} speech chunks ({total_audio_bytes/48000:.1f}s audio).")
    if not tool_calls and not audio_replies:
        print("⚠️ No tool call or audio generated.")

    await task.cancel()
    await runner_task

if __name__ == "__main__":
    asyncio.run(main())
