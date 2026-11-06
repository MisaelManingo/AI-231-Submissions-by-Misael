#!/usr/bin/env python3
"""
Turn Latency Benchmark Instrumentation
Records high-precision timestamps for:
- End of user speech (local Silero VAD)
- First audio chunk received from Gemini Live
- First audio frame rendered on speaker
- Tool-call start & finish
- Barge-in reaction time (user speech onset to playback silent)
Appends results to CSV per turn.
"""

import os
import csv
import time
import threading
from pathlib import Path
from config import BENCHMARK_LOG_PATH


class BenchmarkLogger:
    """Thread-safe turn latency recorder."""
    HEADERS = [
        "turn_id",
        "timestamp_iso",
        "user_speech_end_ts",
        "model_first_audio_ts",
        "speaker_first_audio_ts",
        "eos_to_model_audio_ms",
        "eos_to_speaker_audio_ms",
        "speaker_buffer_ms",
        "tool_name",
        "tool_start_ts",
        "tool_finish_ts",
        "tool_duration_ms",
        "barge_in_detected",
        "barge_in_user_start_ts",
        "barge_in_silent_ts",
        "barge_in_stop_ms"
    ]

    def __init__(self, log_path=BENCHMARK_LOG_PATH):
        self.log_path = Path(log_path)
        self.lock = threading.Lock()
        self.turn_counter = 0

        # Current turn state
        self.current_turn = {}
        self._init_csv()

    def _init_csv(self):
        with self.lock:
            if not self.log_path.exists():
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow(self.HEADERS)

    def start_new_turn(self):
        """Starts a fresh turn context."""
        with self.lock:
            self.turn_counter += 1
            self.current_turn = {
                "turn_id": self.turn_counter,
                "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "user_speech_end_ts": None,
                "model_first_audio_ts": None,
                "speaker_first_audio_ts": None,
                "tool_name": None,
                "tool_start_ts": None,
                "tool_finish_ts": None,
                "tool_duration_ms": 0.0,
                "barge_in_detected": False,
                "barge_in_user_start_ts": None,
                "barge_in_silent_ts": None,
                "barge_in_stop_ms": 0.0,
            }

    def record_user_speech_end(self, ts=None):
        ts = ts or time.perf_counter()
        with self.lock:
            if not self.current_turn:
                self.start_new_turn()
            self.current_turn["user_speech_end_ts"] = ts

    def record_model_first_audio(self, ts=None):
        ts = ts or time.perf_counter()
        with self.lock:
            if self.current_turn and self.current_turn["model_first_audio_ts"] is None:
                self.current_turn["model_first_audio_ts"] = ts

    def record_speaker_first_audio(self, ts=None):
        ts = ts or time.perf_counter()
        with self.lock:
            if self.current_turn and self.current_turn["speaker_first_audio_ts"] is None:
                self.current_turn["speaker_first_audio_ts"] = ts
                # Calculate turn metrics and commit
                self._commit_turn_unlocked()

    def record_tool_start(self, tool_name: str, ts=None):
        ts = ts or time.perf_counter()
        with self.lock:
            if not self.current_turn:
                self.start_new_turn()
            self.current_turn["tool_name"] = tool_name
            self.current_turn["tool_start_ts"] = ts

    def record_tool_finish(self, tool_name: str, ts=None):
        ts = ts or time.perf_counter()
        with self.lock:
            if self.current_turn and self.current_turn["tool_start_ts"]:
                self.current_turn["tool_finish_ts"] = ts
                self.current_turn["tool_duration_ms"] = round(
                    (ts - self.current_turn["tool_start_ts"]) * 1000.0, 2
                )

    def record_barge_in(self, start_ts: float, silent_ts: float):
        with self.lock:
            if not self.current_turn:
                self.start_new_turn()
            self.current_turn["barge_in_detected"] = True
            self.current_turn["barge_in_user_start_ts"] = start_ts
            self.current_turn["barge_in_silent_ts"] = silent_ts
            self.current_turn["barge_in_stop_ms"] = round(
                (silent_ts - start_ts) * 1000.0, 2
            )
            self._commit_turn_unlocked()

    def _commit_turn_unlocked(self):
        """Calculates derived metrics and writes row to CSV."""
        turn = self.current_turn
        if not turn:
            return

        eos = turn.get("user_speech_end_ts")
        model_audio = turn.get("model_first_audio_ts")
        spk_audio = turn.get("speaker_first_audio_ts")

        eos_to_model = round((model_audio - eos) * 1000.0, 2) if (eos and model_audio) else ""
        eos_to_spk = round((spk_audio - eos) * 1000.0, 2) if (eos and spk_audio) else ""
        spk_buf = round((spk_audio - model_audio) * 1000.0, 2) if (model_audio and spk_audio) else ""

        row = [
            turn.get("turn_id"),
            turn.get("timestamp_iso"),
            f"{eos:.6f}" if eos else "",
            f"{model_audio:.6f}" if model_audio else "",
            f"{spk_audio:.6f}" if spk_audio else "",
            eos_to_model,
            eos_to_spk,
            spk_buf,
            turn.get("tool_name") or "",
            f"{turn.get('tool_start_ts'):.6f}" if turn.get("tool_start_ts") else "",
            f"{turn.get('tool_finish_ts'):.6f}" if turn.get("tool_finish_ts") else "",
            turn.get("tool_duration_ms") or "",
            turn.get("barge_in_detected", False),
            f"{turn.get('barge_in_user_start_ts'):.6f}" if turn.get("barge_in_user_start_ts") else "",
            f"{turn.get('barge_in_silent_ts'):.6f}" if turn.get("barge_in_silent_ts") else "",
            turn.get("barge_in_stop_ms") or ""
        ]

        try:
            with open(self.log_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"📊 [BENCHMARK LOGGED]: Turn {turn.get('turn_id')} | EOS -> Spk: {eos_to_spk}ms | Tool: {turn.get('tool_name') or 'None'}")
        except Exception as e:
            print(f"[BENCHMARK CSV ERROR]: {e}")

