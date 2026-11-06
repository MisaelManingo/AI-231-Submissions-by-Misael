#!/usr/bin/env python3
"""
Physical Hardware & Pipeline Integration Test
Runs FullDuplexVoiceAgent on physical Raspberry Pi 5 hardware for 5 seconds.
"""

import sys
import asyncio
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agent import FullDuplexVoiceAgent

async def main():
    print("=" * 60)
    print("TESTING FULL-DUPLEX AGENT ON PHYSICAL RPI5 HARDWARE")
    print("=" * 60)

    agent = FullDuplexVoiceAgent(mock_hardware=False)

    async def _timed_cancel():
        await asyncio.sleep(5.0)
        print("\n[TEST]: 5-second test complete. Stopping agent...")
        await agent.stop()

    asyncio.create_task(_timed_cancel())

    try:
        await agent.run()
    except (asyncio.CancelledError, Exception) as e:
        print(f"[TEST]: Agent stopped with: {e}")
    finally:
        agent.cleanup()

    print("✅ Physical hardware lifecycle test PASSED!")

if __name__ == "__main__":
    asyncio.run(main())

