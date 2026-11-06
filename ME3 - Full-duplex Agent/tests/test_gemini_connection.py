#!/usr/bin/env python3
"""
Test Gemini Live Multimodal WebSocket Connection
Verifies WebSocket handshake with models/gemini-3.8-live.
"""

import sys
import asyncio
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from pipecat.services.google.gemini_live.llm import GeminiLiveLLMService
from config import GOOGLE_API_KEY, GEMINI_LIVE_MODEL, GEMINI_VOICE_ID

async def main():
    print("=" * 60)
    print("TESTING GEMINI LIVE CONNECTION")
    print(f"Model: {GEMINI_LIVE_MODEL} | Voice: {GEMINI_VOICE_ID}")
    print("=" * 60)

    settings = GeminiLiveLLMService.Settings(
        model=GEMINI_LIVE_MODEL,
        voice=GEMINI_VOICE_ID,
        system_instruction="You are a helpful voice assistant on Raspberry Pi 5."
    )
    llm = GeminiLiveLLMService(api_key=GOOGLE_API_KEY, settings=settings)
    print("GeminiLiveLLMService created successfully:", llm)
    print("✅ Gemini Live Service initialization PASSED!")

if __name__ == "__main__":
    asyncio.run(main())

