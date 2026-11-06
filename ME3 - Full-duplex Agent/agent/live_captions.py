#!/usr/bin/env python3
"""
Live Captions tap: streams the SAME mic audio that Gemini Live receives during a user turn to a second,
transcription-only Live session (gemini-3.5-transcribe-live) and forwards its *interim* (speculative) text to the
GUI as a live preview. The conversational model gemini-3.8-live does not emit interim transcripts.

Design / safety:
- Preview only. The authoritative text is still the main session's input_transcription; interim text is never fed
  back to the model.
- feed() is called from Gemini3LiveService._send_user_audio on the asyncio loop: O(1) deque/queue operations only,
  no I/O. All network I/O happens in one worker task + one receiver task (awaited, non-blocking).
- Bounded queue: when the worker is slow or disconnected, the oldest audio is dropped (captions lag, agent never does).
- Audio is sent only while the user is speaking (same audio, same echo gate as the main session), batched into
  100 ms chunks, and audio_stream_end is sent at the end of each utterance (documented "hybrid VAD" finalization).
- Any failure is logged and retried with backoff; it can never raise into the agent.
- Disable entirely with ME3_LIVE_CAPTIONS=0. Model: ME3_CAPTION_MODEL. Languages: ME3_CAPTION_LANGS (comma list,
  default empty = automatic language detection incl. code-switching).
"""

import os
import asyncio
import time
from collections import deque

CAPTIONS_ENABLED = os.getenv("ME3_LIVE_CAPTIONS", "1") != "0"
CAPTION_MODEL = os.getenv("ME3_CAPTION_MODEL", "gemini-3.5-transcribe-live")
CAPTION_LANGS = [c.strip() for c in os.getenv("ME3_CAPTION_LANGS", "").split(",") if c.strip()]
CAPTION_DEBUG = os.getenv("ME3_CAPTION_DEBUG", "0") == "1"   # logs every interim/final event

_CHUNK_BYTES = 3200          # 100 ms @ 16 kHz mono int16 (docs: send ~100 ms chunks)
_PREROLL_FRAMES = 10         # ~320 ms of audio before speech onset (frames are 32 ms)
_QUEUE_MAX = 400             # ~12 s of 32 ms frames; oldest dropped when full


class LiveCaptionTap:
    def __init__(self, api_key: str, publisher_getter):
        self._api_key = api_key
        self._get_publisher = publisher_getter       # callable -> publisher or None (set after construction)
        self._ring = deque(maxlen=_PREROLL_FRAMES)   # recent (audio, rate) while not speaking
        self._queue = deque(maxlen=_QUEUE_MAX)       # items: ("a", bytes, rate) | ("end",)
        self._wake = None                            # asyncio.Event, created in start()
        self._active = False                         # inside an utterance
        self._task = None
        self._recv_task = None
        self._utt_done = False                       # transcribe model finalized this utterance: ignore late interims
        self._fail_count = 0

    # ------------------------------------------------------------------ called from the audio path (no I/O)
    def feed(self, audio: bytes, rate: int, speaking: bool):
        if not audio:
            return
        if speaking:
            if not self._active:
                self._active = True
                self._utt_done = False
                for a, r in self._ring:              # speech onset: include a little audio before it
                    self._queue.append(("a", a, r))
                self._ring.clear()
            self._queue.append(("a", audio, rate))
            if self._wake is not None:
                self._wake.set()
        else:
            if self._active:
                self._active = False
                self._queue.append(("end",))
                if self._wake is not None:
                    self._wake.set()
            self._ring.append((audio, rate))

    # ------------------------------------------------------------------ lifecycle
    def start(self, create_task):
        if self._task is not None and not self._task.done():
            return
        self._wake = asyncio.Event()
        self._task = create_task(self._worker())

    async def stop(self, cancel_task):
        for t in (self._task, self._recv_task):
            if t is not None and not t.done():
                await cancel_task(t)
        self._task = self._recv_task = None

    # ------------------------------------------------------------------ worker
    async def _worker(self):
        from google import genai
        from google.genai import types
        client = genai.Client(api_key=self._api_key)
        config = types.LiveConnectConfig(
            response_modalities=["TEXT"],
            input_audio_transcription=types.AudioTranscriptionConfig(language_codes=CAPTION_LANGS),
        )
        backoff = 2.0
        while True:
            cm = session = None
            try:
                # Connect lazily: wait for the first utterance so idle time costs nothing.
                while not self._queue:
                    self._wake.clear()
                    await self._wake.wait()
                cm = client.aio.live.connect(model=CAPTION_MODEL, config=config)
                session = await cm.__aenter__()
                if self._fail_count:
                    print("📝 [CAPTIONS]: reconnected", flush=True)
                self._fail_count = 0
                backoff = 2.0
                dead = asyncio.Event()
                self._recv_task = asyncio.create_task(self._receiver(session, dead))
                buf = bytearray()
                rate = 16000
                while not dead.is_set():
                    if not self._queue:
                        self._wake.clear()
                        try:
                            await asyncio.wait_for(self._wake.wait(), timeout=60.0)
                        except asyncio.TimeoutError:
                            break                    # idle for a minute: close; reconnect on next speech
                        continue
                    item = self._queue.popleft()
                    if item[0] == "a":
                        buf.extend(item[1])
                        rate = item[2]
                        if len(buf) >= _CHUNK_BYTES:
                            await session.send_realtime_input(
                                audio=types.Blob(data=bytes(buf), mime_type=f"audio/pcm;rate={rate}"))
                            buf.clear()
                    else:  # utterance end: flush the remainder, then ask for immediate finalization
                        if buf:
                            await session.send_realtime_input(
                                audio=types.Blob(data=bytes(buf), mime_type=f"audio/pcm;rate={rate}"))
                            buf.clear()
                        await session.send_realtime_input(audio_stream_end=True)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._fail_count += 1
                if self._fail_count <= 3 or self._fail_count % 10 == 0:
                    print(f"📝 [CAPTIONS]: session error ({type(e).__name__}: {str(e)[:160]}); "
                          f"retry in {backoff:.0f}s (attempt {self._fail_count})", flush=True)
                self._queue.clear()                  # never build a backlog while disconnected
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 60.0)
            finally:
                if self._recv_task is not None:
                    self._recv_task.cancel()
                    self._recv_task = None
                if cm is not None and session is not None:
                    try:
                        await cm.__aexit__(None, None, None)
                    except Exception:
                        pass

    async def _receiver(self, session, dead):
        """Reads transcripts from the caption session and forwards interim text as the GUI live preview."""
        try:
            while True:
                async for resp in session.receive():
                    sc = resp.server_content
                    if not sc:
                        continue
                    pub = self._get_publisher()
                    interim = getattr(sc, "interim_input_transcription", None)
                    if self._active:
                        self._utt_done = False       # user still talking (pause mid-utterance): keep updating
                    if interim and interim.text and not self._utt_done:
                        if CAPTION_DEBUG:
                            print(f"📝 [CAPTIONS interim]: {interim.text!r}", flush=True)
                        if pub:
                            pub.publish_transcript("user", "interim", interim.text)
                    final = sc.input_transcription
                    if final and final.text:
                        if CAPTION_DEBUG:
                            print(f"📝 [CAPTIONS final]: {final.text!r}", flush=True)
                        # Keep the preview showing the finalized caption text until the main session's authoritative
                        # text replaces it, and ignore stragglers so a cleared preview does not reappear.
                        if pub and not self._utt_done:
                            pub.publish_transcript("user", "interim", final.text)
                        if not self._active:
                            self._utt_done = True    # speech really ended: drop straggling interims
        except asyncio.CancelledError:
            raise
        except Exception as e:
            if CAPTION_DEBUG:
                print(f"📝 [CAPTIONS]: receiver ended ({type(e).__name__}: {e})", flush=True)
        finally:
            dead.set()
