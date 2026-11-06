#!/usr/bin/env python3
"""
Test Local Voice Barge-In on LocalVADBargeInProcessor
Verifies that when speaker is active (agent speaking), incoming user speech:
1. Triggers voice barge-in detection
2. Immediately invokes transport.cancel_playback_immediately()
3. Pushes InterruptionFrame downstream
4. Pushes UserStartedSpeakingFrame downstream to begin new user turn
"""

import sys
import asyncio
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import numpy as np
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask
from pipecat.pipeline.runner import PipelineRunner
from pipecat.frames.frames import Frame, InputAudioRawFrame, InterruptionFrame, UserStartedSpeakingFrame
from pipecat.processors.frame_processor import FrameProcessor, FrameDirection
from agent.vad import LocalVADBargeInProcessor


class MockQueue:
    def __init__(self, empty=False):
        self._empty = empty

    def empty(self):
        return self._empty


class MockOutputTransport:
    def __init__(self):
        self.playback_queue = MockQueue(empty=False)


class MockAudioTransport:
    def __init__(self):
        self.is_playing = True
        self.output = MockOutputTransport()
        self.cancelled = False

    def cancel_playback_immediately(self):
        self.cancelled = True
        self.is_playing = False


async def test_barge_in():
    print("=" * 60)
    print("TESTING LOCAL VOICE BARGE-IN DETECTION")
    print("=" * 60)

    transport = MockAudioTransport()
    vad_proc = LocalVADBargeInProcessor(audio_transport=transport)
    received_types = []

    class Sink(FrameProcessor):
        async def process_frame(self, frame: Frame, direction: FrameDirection):
            await super().process_frame(frame, direction)
            received_types.append(type(frame).__name__)

    sink = Sink()
    pipeline = Pipeline([vad_proc, sink])
    task = PipelineTask(pipeline)
    runner = PipelineRunner()

    runner_task = asyncio.create_task(runner.run(task))
    await asyncio.sleep(0.5)

    # 1. Feed quiet/ambient frames while speaker is active: should NOT barge in
    quiet_pcm = (np.random.normal(0, 0.01, 512).astype(np.float32) * 32767).astype(np.int16).tobytes()
    for _ in range(3):
        frame = InputAudioRawFrame(audio=quiet_pcm, sample_rate=16000, num_channels=1)
        await vad_proc.queue_frame(frame, FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.032)

    assert not transport.cancelled, "Quiet frames should NOT trigger barge-in!"
    print("✅ Quiet frames during speaker playback: Echo gate active, zero false barge-ins.")

    # 2. Feed intentional speech frames (multi-harmonic voice simulating loud speech / close proximity)
    t = np.linspace(0, 0.032, 512, endpoint=False)
    voice_samples = 0.50 * np.sin(2 * np.pi * 150 * t) + 0.30 * np.sin(2 * np.pi * 300 * t) + 0.15 * np.sin(2 * np.pi * 450 * t)
    voice_pcm = (np.clip(voice_samples, -0.95, 0.95) * 32767).astype(np.int16).tobytes()

    # 2a. BACKCHANNEL: a short burst (~100 ms, like "yeah") must NOT interrupt the agent
    for _ in range(3):
        frame = InputAudioRawFrame(audio=voice_pcm, sample_rate=16000, num_channels=1)
        await vad_proc.queue_frame(frame, FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.032)
    for _ in range(6):
        frame = InputAudioRawFrame(audio=quiet_pcm, sample_rate=16000, num_channels=1)
        await vad_proc.queue_frame(frame, FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.032)
    assert not transport.cancelled, "Short backchannel burst must NOT trigger barge-in!"
    print("✅ Short burst (backchannel) ignored: agent keeps talking.")

    # 2b. SUSTAINED speech (> BARGE_IN_MIN_MS) must interrupt
    for _ in range(vad_proc.barge_in_min_frames + 6):
        frame = InputAudioRawFrame(audio=voice_pcm, sample_rate=16000, num_channels=1)
        await vad_proc.queue_frame(frame, FrameDirection.DOWNSTREAM)
        await asyncio.sleep(0.032)

    await asyncio.sleep(0.2)
    await runner.cancel()
    await runner_task

    assert transport.cancelled, "Voice frames should trigger transport cancellation!"
    assert "InterruptionFrame" in received_types, "InterruptionFrame MUST be pushed downstream!"
    assert "UserStartedSpeakingFrame" in received_types, "UserStartedSpeakingFrame MUST be pushed downstream!"
    assert vad_proc.user_speaking, "vad_proc.user_speaking MUST be True after barge-in!"

    print("✅ Voice barge-in triggered successfully!")
    print(f"   - Speaker cancelled: {transport.cancelled}")
    print(f"   - Received frames: {received_types}")
    print(f"   - Agent switched to user_speaking: {vad_proc.user_speaking}")
    print("=" * 60)
    print("ALL VOICE BARGE-IN ASSERTIONS PASSED!")


if __name__ == "__main__":
    asyncio.run(test_barge_in())
