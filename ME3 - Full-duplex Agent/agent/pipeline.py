#!/usr/bin/env python3
"""
Full-Duplex Voice Agent Pipeline Coordinator
Integrates:
- GMarkAudioTransport (48kHz ALSA <-> 16kHz Pipecat conversion)
- LocalVADBargeInProcessor (Silero ONNX local turn detection & echo-gated barge-in)
- GeminiLiveLLMService (Speech-to-speech with models/gemini-3.8-live)
- Local MCP Server & Function Calling Bridge
- Hardware Manager (20x4 I2C LCD status, TM1637 clock/timer, WS2813 LED strip, RGB LED, Buzzer)
- Turn Latency Benchmark Logger
"""

import os
import sys
import asyncio
import time
from pathlib import Path
import numpy as np
import soundfile as sf

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.task import PipelineTask, PipelineParams
from pipecat.pipeline.runner import PipelineRunner
from pipecat.processors.aggregators.llm_response_universal import (
    LLMContext, LLMContextAggregatorPair, LLMUserAggregatorParams, UserTurnStrategies
)
from pipecat.turns.user_start.external_user_turn_start_strategy import ExternalUserTurnStartStrategy
from pipecat.turns.user_stop.external_user_turn_stop_strategy import ExternalUserTurnStopStrategy
from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService, GeminiModalities, GeminiVADParams
from pipecat.adapters.llm.schemas.function_schema import FunctionSchema
from pipecat.frames.frames import TTSAudioRawFrame, TTSTextFrame, LLMTextFrame
from pipecat.processors.frame_processor import FrameDirection

from config import (
    GOOGLE_API_KEY, GEMINI_LIVE_MODEL, GEMINI_VOICE_ID, GEMINI_LANGUAGE, BILINGUAL,
    SYSTEM_INSTRUCTION, AUDIO_MODEL_RATE, GUI_SOCKET_PATH, SAVE_SENT_AUDIO, ATTENTION_ENABLED
)
from hardware import HardwareManager
from mcp_server import create_mcp_server
from mcp_server.client_bridge import get_mcp_function_schemas, bridge_mcp_to_pipecat
from benchmark import BenchmarkLogger
from .audio_transport import GMarkAudioSystem
from .vad import LocalVADBargeInProcessor
from .state_publisher import AgentStatePublisher
from .live_captions import LiveCaptionTap, CAPTIONS_ENABLED


class Gemini3LiveService(GeminiLiveLLMService):
    """
    Subclasses GeminiLiveLLMService to explicitly trigger turn completion
    for Gemini 3.x Live Multimodal API:
    1. Handles natural streaming audio turn detection without injecting premature empty turns.
    2. When a tool call result is returned, sends turn_complete + realtime input nudge so Gemini 3.x
       synthesizes spoken confirmation instead of remaining silent in thinking state.
    3. Handles idle fallback and graceful recovery during interruptions.
    """
    def __init__(self, *args, hw_mgr=None, audio_system=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.hw_mgr = hw_mgr
        self.audio_system = audio_system
        self._current_turn_pcm = bytearray()
        self._turn_index = 0
        # ATTENTION GATE: set when the model called ignore_speech (speech not directed at Berry). While set, this
        # turn's spoken/text reply is dropped so the agent stays silent. Cleared when the user speaks again.
        self._ignore_turn = False
        # GUI transcript overlay: set by FullDuplexVoiceAgent after the publisher exists. publish_transcript() is a
        # non-blocking queue append (no I/O on the event loop). Remove this + the hooks below to revert.
        self.publisher = None
        # GUI live preview of user speech (interim_input_transcription, speculative; observer only, never fed back
        # to the model). ME3_GUI_INTERIM=0 disables. First few raw events are logged to learn the payload semantics.
        self._interim_enabled = os.getenv("ME3_GUI_INTERIM", "1") != "0"
        self._interim_logged = 0
        # IDLE KEEPALIVE: ME3_KEEPALIVE_S seconds of no client/server traffic => send 100 ms of digital silence.
        # 0 disables. Task lives and dies with the connection task handler (see _connection_task_handler).
        self._keepalive_s = float(os.getenv("ME3_KEEPALIVE_S", "45"))
        self._last_io = time.monotonic()
        # LIVE CAPTIONS: second transcription-only session feeding the GUI preview (agent/live_captions.py).
        # ME3_LIVE_CAPTIONS=0 disables; remove this block + the 3 hooks below to revert.
        self._captions = LiveCaptionTap(GOOGLE_API_KEY, lambda: self.publisher) if CAPTIONS_ENABLED else None
        # DIAGNOSTIC ONLY: ME3_AUDIO_TRACE=1 prints frames received by the Gemini service per turn
        self._trace = os.getenv("ME3_AUDIO_TRACE", "0") == "1"
        self._tr_frames = 0
        self._tr_zero_frames = 0
        self._tr_not_sent = 0
        self._tr_t0 = 0.0
        self.recordings_dir = BASE_DIR / "tests" / "recordings"
        self.recordings_dir.mkdir(parents=True, exist_ok=True)

    async def _connection_task_handler(self, config):
        from google.genai.types import AudioTranscriptionConfig, AudioTranscriptionConfigMode
        # Strict English language lock with SMART neural disfluency removal and comprehensive smart-home vocabulary.
        # Eliminates raw phonetic artifacts (e.g. 'Make Die Blue', 'gelatin', 'MC') and guarantees coherent English transcription.
        config.input_audio_transcription = AudioTranscriptionConfig(
            language_codes=["en-US", "fil-PH"] if BILINGUAL else ["en-US"],
            mode=AudioTranscriptionConfigMode.SMART,
            custom_vocabulary=[
                # Lights & Colors
                "turn on the lights", "turn off the lights", "lights on", "lights off",
                "Berry", "Hey Berry", "Berry,", "okay Berry",
                "turn on the light", "turn off the light", "turn light on", "turn lights on",
                "red", "green", "blue", "white", "yellow", "cyan", "purple",
                "make the lights blue", "make the lights green", "make the lights red",
                "make the lights white", "make the lights yellow", "make the lights cyan", "make the lights purple",
                "turn the lights blue", "turn the lights green", "turn the lights red",
                "turn the lights white", "turn the lights yellow", "turn the lights cyan", "turn the lights purple",
                "set lights to blue", "set lights to green", "set lights to red",
                "set lights to white", "set lights to yellow", "set lights to cyan", "set lights to purple",
                "make it blue", "make it green", "make it red", "make it white", "make it yellow", "make it purple",
                # Media & Music
                "play music", "pause music", "stop music", "resume music",
                "next track", "previous track", "volume up", "volume down", "music", "song", "track",
                # Weather & Environmental
                "what is the weather", "weather in Manila", "what is the weather in Manila", "temperature",
                "thermostat", "setpoint", "aircon", "cooler", "heater",
                # System & Clock
                "what time is it", "what is the date", "tell me the time", "tell me a joke",
                "why is the sky blue",
                "set timer", "cancel timer", "stop timer", "reset timer",
                "set alarm", "cancel alarm", "stop alarm",
                # Productivity & Telephony
                "add reminder", "list reminders", "clear reminder",
                "make phone call", "send sms", "create simple website",
                "run shell command", "execute bash"
            ] + ([  # bilingual mode: a few Filipino smart-home phrases to bias recognition
                "buksan ang ilaw", "patayin ang ilaw", "kumusta", "salamat", "anong oras na",
                "ano ang panahon", "magpatugtog ng musika", "itigil ang musika", "pakilakasan", "pakihinaan",
            ] if BILINGUAL else [])
        )
        vad_status = "DISABLED (Client Local VAD Active)" if self._vad_disabled else "ENABLED (Server Cloud VAD Active)"
        print(f"🔒 [GEMINI VAD STATUS]: Server VAD is {vad_status}.", flush=True)
        print("🌐 [GEMINI LIVE]: Speech recognition locked to English (SMART mode with full smart-home vocabulary).", flush=True)
        # Keepalive runs only while this connection's handler runs, so reconnects get a fresh one automatically.
        ka_task = self.create_task(self._keepalive_loop()) if self._keepalive_s > 0 else None
        if self._captions:
            self._captions.start(self.create_task)
        try:
            await super()._connection_task_handler(config)
        finally:
            if self._captions:
                await self._captions.stop(self.cancel_task)
            if ka_task:
                await self.cancel_task(ka_task)

    async def _keepalive_loop(self):
        """Idle keepalive: tiny silent audio chunk when nothing has moved on the session for a while.
        Skipped while the user is speaking, input is paused (echo gate / listen probe), or not connected.
        NOTE: in local-VAD mode Gemini discards audio outside an activity window, so whether this actually prevents
        the server-side idle close must be verified live; it is harmless if ignored."""
        from google.genai.types import Blob
        silence = b"\x00" * 3200  # 100 ms @ 16 kHz mono int16
        while True:
            await asyncio.sleep(5)
            try:
                if (time.monotonic() - self._last_io < self._keepalive_s
                        or self._user_is_speaking or self._audio_input_paused or self._disconnecting
                        or not self._session or not self._ready_for_realtime_input):
                    continue
                await self._session.send_realtime_input(audio=Blob(data=silence, mime_type="audio/pcm;rate=16000"))
                self._last_io = time.monotonic()
                print("💓 [KEEPALIVE]: sent 100 ms silence (idle)", flush=True)
            except asyncio.CancelledError:
                raise
            except Exception as e:  # never let the keepalive affect the agent
                self._last_io = time.monotonic()  # back off a full interval instead of retrying every 5 s
                print(f"💓 [KEEPALIVE]: send failed ({type(e).__name__}); leaving recovery to the normal handler", flush=True)

    async def _handle_server_message(self, message):
        self._last_io = time.monotonic()  # any server traffic counts as activity for the keepalive
        # Observer only: forward interim (speculative) user transcription to the GUI. Pipecat ignores this field.
        if self._interim_enabled and self.publisher:
            sc = message.server_content
            interim = getattr(sc, "interim_input_transcription", None) if sc else None
            if interim and interim.text:
                if self._interim_logged < 8:
                    self._interim_logged += 1
                    print(f"👂 [INTERIM RAW #{self._interim_logged}]: {interim.text!r}", flush=True)
                self.publisher.publish_transcript("user", "interim", interim.text)
        await super()._handle_server_message(message)

    async def _send_user_audio(self, frame):
        if self._trace and self._user_is_speaking and frame.audio:
            self._tr_frames += 1
            if not any(frame.audio):
                self._tr_zero_frames += 1
            if self._audio_input_paused or self._disconnecting or not self._session or not self._ready_for_realtime_input:
                self._tr_not_sent += 1
        if SAVE_SENT_AUDIO and self._user_is_speaking and frame.audio:  # debug dump only (ME3_SAVE_SENT_AUDIO=1)
            self._current_turn_pcm.extend(frame.audio)
        self._last_io = time.monotonic()
        if self._captions and frame.audio:  # O(1) tap: queue the same audio for the caption session
            self._captions.feed(frame.audio, frame.sample_rate, self._user_is_speaking)
        await super()._send_user_audio(frame)

    async def push_frame(self, frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        if (self._ignore_turn and direction == FrameDirection.DOWNSTREAM
                and isinstance(frame, (TTSAudioRawFrame, TTSTextFrame, LLMTextFrame))):
            return  # attention gate: speech was not for us, do not answer it
        await super().push_frame(frame, direction)

    async def _handle_user_started_speaking(self, frame):
        self._ignore_turn = False
        self._turn_index += 1
        self._current_turn_pcm = bytearray()
        if self._trace:
            if self._tr_frames:
                print(f"[TRACE LLM] previous turn never got UserStoppedSpeaking: received {self._tr_frames} frames, {self._tr_zero_frames} all-zero", flush=True)
            self._tr_frames = self._tr_zero_frames = self._tr_not_sent = 0
            self._tr_t0 = time.perf_counter()
            print(f"[TRACE LLM] turn {self._turn_index}: UserStartedSpeaking received (ready={self._ready_for_realtime_input})", flush=True)
        await super()._handle_user_started_speaking(frame)

    async def _save_turn_recording(self, turn_idx: int, pcm_bytes: bytes):
        try:
            samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
            dur = len(samples) / 16000.0
            peak = float(np.max(np.abs(samples))) if len(samples) > 0 else 0.0
            rms = float(np.sqrt(np.mean(samples ** 2))) if len(samples) > 0 else 0.0

            out_file = self.recordings_dir / f"live_sent_turn_{turn_idx}.wav"
            await asyncio.to_thread(sf.write, str(out_file), samples, 16000)
            print(f"💾 [AUDIO AUDIT]: Saved EXACT audio sent to Gemini -> tests/recordings/{out_file.name} ({dur:.2f}s | Peak: {peak:.3f} | RMS: {rms:.3f})", flush=True)
        except Exception as e:
            print(f"⚠️ [AUDIO AUDIT]: Failed to save turn audio: {e}", flush=True)

    async def _push_user_transcription(self, text: str, *, result=None):
        print(f"\n🗣️  [USER HEARD]: \"{text}\"", flush=True)
        if self.publisher:
            self.publisher.publish_transcript("user", "add", text)
        await super()._push_user_transcription(text, result=result)

    async def _handle_msg_output_transcription(self, message):
        if self._ignore_turn:
            return
        if message.server_content and message.server_content.output_transcription:
            text = message.server_content.output_transcription.text
            if text:
                print(f"🤖 [AGENT SAYS]: {text}", flush=True)
                if self.publisher:
                    self.publisher.publish_transcript("ai", "append", text)
        await super()._handle_msg_output_transcription(message)

    async def _handle_user_stopped_speaking(self, frame):
        if self._trace:
            print(f"[TRACE LLM] turn {self._turn_index}: received {self._tr_frames} frames ({self._tr_frames*32}ms), "
                  f"{self._tr_zero_frames} all-zero, {self._tr_not_sent} NOT sent (paused/not ready), wall={(time.perf_counter()-self._tr_t0)*1000:.0f}ms", flush=True)
            self._tr_frames = self._tr_zero_frames = self._tr_not_sent = 0
        await super()._handle_user_stopped_speaking(frame)
        if SAVE_SENT_AUDIO and self._current_turn_pcm:
            pcm_data = bytes(self._current_turn_pcm)
            self._current_turn_pcm = bytearray()
            asyncio.create_task(self._save_turn_recording(self._turn_index, pcm_data))

    async def _tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        tool_result_message: dict,
        *,
        is_final: bool = True,
        spoken: bool = True,
    ):
        await super()._tool_result(
            tool_call_id,
            tool_name,
            tool_result_message,
            is_final=is_final,
            spoken=spoken,
        )

    async def _handle_msg_turn_complete(self, message):
        await super()._handle_msg_turn_complete(message)
        if self.publisher:
            self.publisher.publish_transcript("ai", "end")
        # Safety fallback: if turn completes and audio system is not playing,
        # restore hardware display back to idle after a grace period.
        if self.hw_mgr and self.audio_system:
            async def _check_idle():
                await asyncio.sleep(1.0)
                if not self.audio_system.is_playing:
                    self.hw_mgr.on_idle()
            asyncio.create_task(_check_idle())

    async def _handle_interruption(self):
        self._ignore_turn = False
        if self.publisher:
            self.publisher.publish_transcript("ai", "interrupt")
        self._current_turn_pcm = bytearray()
        await super()._handle_interruption()
        if self.hw_mgr:
            try:
                self.hw_mgr.on_idle()
            except Exception:
                pass


class FullDuplexVoiceAgent:
    """Coordinates the full-duplex speech-to-speech AI assistant."""
    def __init__(self, mock_hardware=False):
        print("=" * 65)
        print("INITIALIZING ME3 FULL-DUPLEX VOICE AGENT")
        print(f"Model: {GEMINI_LIVE_MODEL} | Voice: {GEMINI_VOICE_ID}")
        print("=" * 65)

        # 1. Hardware manager & Benchmark logger
        self.hw_mgr = HardwareManager(mock=mock_hardware)
        self.benchmark = BenchmarkLogger()

        # 2. Local MCP Server
        self.mcp = create_mcp_server(self.hw_mgr)
        self.tool_schemas = get_mcp_function_schemas(self.mcp)
        if ATTENTION_ENABLED:
            # Not an MCP tool: a pipeline-level tool the model calls instead of replying to speech not meant for it.
            self.tool_schemas.append(FunctionSchema(
                name="ignore_speech",
                description=("Call this INSTEAD of replying when the speech you just heard is not directed at you "
                             "(people talking to each other, self-talk, TV/music, background). Produces no speech."),
                properties={},
                required=[],
            ))

        # 3. Gemini Live LLM Service (silent on boot: no unprompted greeting)
        # Client-driven VAD (vad.disabled=True): Disables cloud-side auto activity detection
        # so local Silero VAD on Pi 5 drives turn starts (activity_start) and turn stops (activity_end).
        # user_audio_preroll_secs=0.0: Disables Pipecat's 16KB unchunked ambient noise dump so
        # LocalVADBargeInProcessor handles smooth, continuous audio frame delivery from speech onset.
        settings = GeminiLiveLLMService.Settings(
            model=GEMINI_LIVE_MODEL,
            voice=GEMINI_VOICE_ID,
            modalities=GeminiModalities.AUDIO,
            language=None if BILINGUAL else GEMINI_LANGUAGE,  # bilingual: do not pin speech_config.language_code
            system_instruction=SYSTEM_INSTRUCTION,
            vad=GeminiVADParams(disabled=True)
        )
        self.llm = Gemini3LiveService(
            api_key=GOOGLE_API_KEY,
            settings=settings,
            tools=self.tool_schemas,
            inference_on_context_initialization=False,
            user_audio_preroll_secs=0.0,
            hw_mgr=self.hw_mgr
        )

        # 4. Bridge MCP Tools to Gemini Live function calling
        bridge_mcp_to_pipecat(self.mcp, self.llm, self.benchmark, self.hw_mgr)
        if ATTENTION_ENABLED:
            async def _ignore_speech(params):
                self.llm._ignore_turn = True
                self.publisher.publish_transcript("ai", "ignore")  # GUI: erase any AI fragments already shown
                self.publisher.publish_transcript("user", "ignore")  # GUI: dim this turn's user lines
                print("🙈 [ATTENTION]: speech not directed at Berry - staying silent.", flush=True)
                try:
                    self.hw_mgr.on_idle()
                except Exception:
                    pass
                await params.result_callback("ignored")
            self.llm.register_function("ignore_speech", _ignore_speech)

        # 5. Audio transport & Local VAD processor
        self.audio_system = GMarkAudioSystem(
            hardware_manager=self.hw_mgr,
            benchmark_logger=self.benchmark
        )
        self.hw_mgr.music_player = self.audio_system.music_player
        self.llm.audio_system = self.audio_system
        self.vad_processor = LocalVADBargeInProcessor(
            audio_transport=self.audio_system,
            hardware_manager=self.hw_mgr,
            benchmark_logger=self.benchmark
        )

        # 6. GUI State & Audio Level Publisher (Decoupled IPC)
        self.publisher = AgentStatePublisher(socket_path=GUI_SOCKET_PATH)
        self.hw_mgr.publisher = self.publisher
        self.llm.publisher = self.publisher  # GUI transcript overlay hooks
        self.vad_processor.publisher = self.publisher
        self.audio_system.output.publisher = self.publisher

        self.runner = None
        self.task = None

    async def run(self):
        """Starts the full-duplex agent loop."""
        loop = asyncio.get_running_loop()

        # Start hardware audio streams
        await self.audio_system.start(loop)

        # Configure LLM context with ExternalUserTurn strategies
        # This completely disables Pipecat's aggressive LocalSmartTurnV3Analyzer ONNX model,
        # ensuring our calibrated LocalVADBargeInProcessor (Silero VAD) exclusively controls turn start and stop.
        context = LLMContext()
        user_strategies = UserTurnStrategies(
            start=[ExternalUserTurnStartStrategy()],
            stop=[ExternalUserTurnStopStrategy(wait_for_transcript=False)]
        )
        user_params = LLMUserAggregatorParams(
            user_turn_strategies=user_strategies
        )
        user_aggregator, assistant_aggregator = LLMContextAggregatorPair(
            context,
            user_params=user_params
        )
        self.llm._context = context

        # Build Pipecat pipeline: Audio In -> Local Silero VAD -> Context Aggregator -> Gemini Live -> Audio Out -> Assistant Aggregator
        pipeline = Pipeline([
            self.audio_system.input,
            self.vad_processor,
            user_aggregator,
            self.llm,
            self.audio_system.output,
            assistant_aggregator
        ])

        # Wire hardware button callback for alarm silence + instant hardware barge-in
        def on_hardware_button():
            self.hw_mgr.alarm_timer.silence_alert()
            if self.audio_system.is_playing:
                print("🔘 [BUTTON]: Hardware barge-in button pressed! Interrupting playback.")
                self.audio_system.cancel_playback_immediately()
                if loop and loop.is_running():
                    from pipecat.processors.frame_processor import FrameDirection
                    from pipecat.frames.frames import InterruptionFrame
                    asyncio.run_coroutine_threadsafe(
                        self.vad_processor.push_frame(InterruptionFrame(), FrameDirection.DOWNSTREAM),
                        loop
                    )

        if hasattr(self.hw_mgr.alarm_timer, "button") and self.hw_mgr.alarm_timer.button:
            self.hw_mgr.alarm_timer.button.set_callback(on_hardware_button)

        params = PipelineParams(
            audio_in_sample_rate=AUDIO_MODEL_RATE,
            audio_out_sample_rate=AUDIO_MODEL_RATE
        )

        self.task = PipelineTask(pipeline, params=params)
        self.runner = PipelineRunner()

        self.hw_mgr.on_idle()
        self.publisher.start(loop)
        print("\n" + "=" * 65)
        print("🎙️  AGENT ONLINE: Always-listening, full-duplex, interruptible!")
        print("Speak naturally into the G-MARK Micro Go microphone.")
        print("Press Ctrl+C to terminate.")
        print("=" * 65 + "\n")

        try:
            await self.runner.add_workers(self.task)
            await self.runner.run()
        except asyncio.CancelledError:
            pass
        finally:
            self.cleanup()

    async def stop(self):
        """Cancels runner and shuts down pipeline."""
        if self.runner:
            await self.runner.cancel()

    def cleanup(self):
        print("\n[AGENT]: Stopping pipeline & cleaning up hardware...")
        if hasattr(self, "publisher") and self.publisher:
            self.publisher.close()
        if hasattr(self, "audio_system") and self.audio_system:
            self.audio_system.close()
        if self.hw_mgr:
            self.hw_mgr.cleanup()
        print("[AGENT]: Shutdown complete.")
