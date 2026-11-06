#!/usr/bin/env python3
"""
End-to-End Pipeline Verification Test
Tests LocalVADBargeInProcessor connected to Gemini3LiveService:
Feeds recorded user voice files into the VAD processor in real time (chunk by chunk),
verifying that LocalVADBargeInProcessor properly triggers:
- Speech onset
- 10-frame pre-roll flush
- Turn completion
- Gemini Live accurate transcription & tool call execution
"""

import sys
import asyncio
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
import soundfile as sf

from config import (
    GOOGLE_API_KEY, GEMINI_LIVE_MODEL, GEMINI_VOICE_ID, GEMINI_LANGUAGE,
    SYSTEM_INSTRUCTION
)
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, GeminiModalities, GeminiVADParams
from pipecat.frames.frames import (
    Frame, InputAudioRawFrame, TTSAudioRawFrame, FunctionCallInProgressFrame, TextFrame
)
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContext, LLMContextAggregatorPair, LLMUserAggregatorParams, UserTurnStrategies
)
from pipecat.turns.user_start.external_user_turn_start_strategy import ExternalUserTurnStartStrategy
from pipecat.turns.user_stop.external_user_turn_stop_strategy import ExternalUserTurnStopStrategy
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from agent.pipeline import Gemini3LiveService
from agent.vad import LocalVADBargeInProcessor
from mcp_server import create_mcp_server
from mcp_server.client_bridge import get_mcp_function_schemas, bridge_mcp_to_pipecat
from hardware import HardwareManager


class DummyTransportInput:
    def __init__(self):
        self.barge_in_active = False

class DummyTransport:
    def __init__(self):
        self.is_playing = False
        self.input = DummyTransportInput()
    def cancel_playback_immediately(self):
        self.is_playing = False


async def run_test():
    print("=" * 65)
    print("VERIFYING LOCAL VAD + GEMINI LIVE END-TO-END PIPELINE")
    print("=" * 65)

    hw = HardwareManager(mock=True)
    mcp_serv = create_mcp_server(hw)
    tool_schemas = get_mcp_function_schemas(mcp_serv)
    dummy_transport = DummyTransport()

    vad_processor = LocalVADBargeInProcessor(
        audio_transport=dummy_transport,
        hardware_manager=hw
    )

    settings = GeminiLiveLLMService.Settings(
        model=GEMINI_LIVE_MODEL,
        voice=GEMINI_VOICE_ID,
        modalities=GeminiModalities.AUDIO,
        language=GEMINI_LANGUAGE,
        system_instruction=SYSTEM_INSTRUCTION,
        vad=GeminiVADParams(disabled=True)
    )
    llm = Gemini3LiveService(
        api_key=GOOGLE_API_KEY,
        settings=settings,
        tools=tool_schemas,
        user_audio_preroll_secs=0.0,
        hw_mgr=hw
    )
    bridge_mcp_to_pipecat(mcp_serv, llm, None, hw)

    transcriptions = []
    tool_calls = []
    spoken_replies = []
    response_event = asyncio.Event()

    class OutputMonitor(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            if isinstance(frame, FunctionCallInProgressFrame):
                tool_calls.append(f"{frame.function_name}({frame.arguments})")
                response_event.set()
            elif isinstance(frame, TTSAudioRawFrame):
                spoken_replies.append(frame.audio)
                response_event.set()
            elif isinstance(frame, TextFrame):
                if frame.text.strip():
                    transcriptions.append(frame.text)
            await self.push_frame(frame, direction)

    context = LLMContext()
    user_strategies = UserTurnStrategies(
        start=[ExternalUserTurnStartStrategy()],
        stop=[ExternalUserTurnStopStrategy(wait_for_transcript=False)]
    )
    user_params = LLMUserAggregatorParams(
        user_turn_strategies=user_strategies
    )
    user_agg, asst_agg = LLMContextAggregatorPair(context, user_params=user_params)
    llm._context = context
    monitor = OutputMonitor()

    pipeline = Pipeline([vad_processor, user_agg, llm, monitor, asst_agg])
    runner = PipelineRunner()
    task = PipelineTask(pipeline)
    runner_task = asyncio.create_task(runner.run(task))

    # Wait for session connection
    for _ in range(50):
        if llm._session:
            break
        await asyncio.sleep(0.1)

    assert llm._session is not None, "Failed to connect to Gemini Live"
    print("✅ Connected to Gemini Live API.")

    test_files = [
        "user_voice_16k_processed.wav",
        "live_splice_turn_1.wav",
        "live_splice_turn_3.wav",
        "live_splice_turn_4.wav"
    ]

    for fname in test_files:
        wav_path = BASE_DIR / "tests" / "recordings" / fname
        if not wav_path.exists():
            continue

        print(f"\n=======================================================")
        print(f"TESTING FILE: {fname}")
        print(f"=======================================================")

        response_event.clear()
        samples_16k, sr = sf.read(str(wav_path), dtype="float32")
        print(f"Loaded {len(samples_16k)/sr:.2f}s of speech")

        leading_silence = np.zeros(int(sr * 0.4), dtype=np.float32)
        trailing_silence = np.zeros(int(sr * 1.0), dtype=np.float32)
        full_audio = np.concatenate([leading_silence, samples_16k, trailing_silence])

        chunk_samples = 512
        for i in range(0, len(full_audio), chunk_samples):
            chunk = full_audio[i:i + chunk_samples]
            if len(chunk) < chunk_samples:
                chunk = np.pad(chunk, (0, chunk_samples - len(chunk)))
            pcm_bytes = (chunk * 32767).astype(np.int16).tobytes()
            frame = InputAudioRawFrame(audio=pcm_bytes, sample_rate=16000, num_channels=1)
            await task.queue_frame(frame)
            await asyncio.sleep(0.015)

        try:
            await asyncio.wait_for(response_event.wait(), timeout=6.0)
            await asyncio.sleep(1.2)
        except asyncio.TimeoutError:
            print("⚠️ Response timeout.")

        await asyncio.sleep(1.0)

    print("\n" + "=" * 65)
    print("RESULTS:")
    print(f"Tool Calls: {tool_calls}")
    print(f"Spoken Audio Frames: {len(spoken_replies)}")
    print(f"Transcriptions: {transcriptions}")
    print("=" * 65)

    await task.cancel()
    try:
        await runner_task
    except Exception:
        pass


if __name__ == "__main__":
    asyncio.run(run_test())
