#!/usr/bin/env python3
"""
Agent State & Audio Level Publisher for ME3 Full-Duplex Agent
Publishes lightweight JSON datagrams over a local Unix domain datagram socket
at 25-30 Hz to decouple GUI and external consumers from the agent pipeline.

Key Guarantees:
- Runs outside PortAudio audio callbacks in a dedicated asyncio task.
- Non-blocking socket sends: if the GUI is not running or crashes, messages are dropped instantly.
- Zero impact on audio latency or frame arrival intervals.
- Swappable interface: socket can be consumed by Pygame, WebSockets, or HTTP SSE.
"""

import os
import time
import json
import socket
import asyncio
from collections import deque
from pathlib import Path
from typing import Optional


# Transcript overlay flag: ME3_GUI_TRANSCRIPT=0 disables publishing (the GUI reads the same flag to hide the overlay).
TRANSCRIPT_ENABLED = os.getenv("ME3_GUI_TRANSCRIPT", "1") != "0"
_TR_INTERIM_CHARS = 200  # interim text is replace-semantics: sent whole (<=800 B), never split
_TR_MAX_CHARS = 120  # <=480 UTF-8 bytes/datagram (sent with ensure_ascii=False), well under the GUI's 2048-byte recv buffer


class AgentStatePublisher:
    """Publishes agent state and audio level to a local Unix datagram socket."""

    def __init__(self, socket_path: Path):
        self.socket_path = socket_path
        self._current_state = "idle"  # "idle", "listening", "thinking", "speaking"
        self._mic_level = 0.0
        self._speaker_level = 0.0
        self._sock: Optional[socket.socket] = None
        self._publish_task: Optional[asyncio.Task] = None
        self._running = False
        # Transcript events queued by pipeline hooks; drained by _publish_loop (bounded: oldest dropped, never blocks)
        self._tr_queue = deque(maxlen=64)

    def publish_transcript(self, who: str, op: str, text: str = ""):
        """O(1), non-blocking: just queues an event. No socket I/O here (called on the asyncio event loop)."""
        if TRANSCRIPT_ENABLED and self._running:
            self._tr_queue.append((who, op, text))

    def _drain_transcripts(self):
        """Builds datagrams from queued events; consecutive AI 'append' fragments are coalesced into one."""
        out = []
        while self._tr_queue:
            who, op, text = self._tr_queue.popleft()
            if out and op == "append" and out[-1]["op"] == "append" and out[-1]["who"] == who:
                out[-1]["text"] += text
                continue
            if op == "interim":  # speculative preview: newest replaces older, keep only the tail
                text = text[-_TR_INTERIM_CHARS:]
                if out and out[-1]["op"] == "interim" and out[-1]["who"] == who:
                    out[-1]["text"] = text
                    continue
            out.append({"type": "tr", "who": who, "op": op, "text": text})
        msgs = []
        for m in out:  # split oversized text by characters (never mid-codepoint) so datagrams stay small
            t = m["text"]
            if m["op"] == "interim" or len(t) <= _TR_MAX_CHARS:
                msgs.append(m)
                continue
            for i in range(0, len(t), _TR_MAX_CHARS):
                msgs.append({**m, "text": t[i:i + _TR_MAX_CHARS]})
        return msgs

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        """Initializes non-blocking datagram socket and starts 25 Hz background task."""
        if self._running:
            return

        self._running = True
        try:
            self._sock = socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM)
            self._sock.setblocking(False)
        except Exception as e:
            print(f"[STATE PUBLISHER]: Warning: failed to create AF_UNIX socket: {e}", flush=True)
            self._sock = None

        if loop is None:
            loop = asyncio.get_running_loop()
        self._publish_task = loop.create_task(self._publish_loop())
        print(f"[STATE PUBLISHER]: State & Level publisher active -> {self.socket_path}", flush=True)

    def set_state(self, state: str, level: Optional[float] = None, rms: Optional[float] = None, **kwargs):
        """Updates the state and optionally sets audio level."""
        self._current_state = state
        lvl = level if level is not None else rms
        if lvl is not None:
            if state == "listening":
                self._mic_level = max(0.0, min(float(lvl), 1.0))
            elif state == "speaking":
                self._speaker_level = max(0.0, min(float(lvl), 1.0))

        if state in ("idle", "thinking"):
            self._mic_level = 0.0
            self._speaker_level = 0.0

    def update_mic_level(self, rms: float):
        """Updates live incoming microphone level."""
        self._mic_level = max(0.0, min(float(rms), 1.0))
        if self._current_state != "listening":
            self._current_state = "listening"

    def update_speaker_level(self, rms: float):
        """Updates live outgoing speaker level with peak hold."""
        val = max(0.0, min(float(rms), 1.0))
        self._speaker_level = max(self._speaker_level, val)
        self._current_state = "speaking"

    async def _publish_loop(self):
        """Publishes state + level at 25 Hz (40 ms cadence)."""
        sock_str = str(self.socket_path)
        while self._running:
            try:
                # Calculate active level based on state with smooth analog decay
                if self._current_state == "speaking":
                    level = self._speaker_level
                    # Decay speaker level smoothly across chunks
                    self._speaker_level *= 0.88
                    if self._speaker_level < 0.005:
                        self._speaker_level = 0.0
                elif self._current_state == "listening":
                    level = self._mic_level
                    self._mic_level *= 0.88
                    if self._mic_level < 0.005:
                        self._mic_level = 0.0
                else:
                    level = 0.0

                if self._sock and self.socket_path.exists():
                    payload = json.dumps({
                        "state": self._current_state,
                        "level": round(float(level), 3),
                        "t": time.time()
                    }).encode("utf-8")
                    try:
                        self._sock.sendto(payload, sock_str)
                    except (BlockingIOError, ConnectionRefusedError, FileNotFoundError, OSError):
                        pass

                # Transcript events (same non-blocking sendto; dropped silently if GUI is absent/busy)
                if self._sock and self._tr_queue:
                    for m in self._drain_transcripts():
                        try:
                            self._sock.sendto(json.dumps(m, ensure_ascii=False).encode("utf-8"), sock_str)
                        except (BlockingIOError, ConnectionRefusedError, FileNotFoundError, OSError):
                            pass
                elif self._tr_queue:
                    self._tr_queue.clear()

                await asyncio.sleep(0.04)  # 25 Hz
            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(0.04)

    def close(self):
        """Shuts down publisher and releases socket resources."""
        self._running = False
        if self._publish_task and not self._publish_task.done():
            self._publish_task.cancel()
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None
        print("[STATE PUBLISHER]: Publisher closed.", flush=True)

