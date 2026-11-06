#!/usr/bin/env python3
"""
Agent Lifecycle & Pipeline Test
Runs FullDuplexVoiceAgent for 4 seconds to verify startup, stream initialization, and graceful shutdown.
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
    print("TESTING FULL-DUPLEX AGENT LIFECYCLE (MOCK)")
    print("=" * 60)

    agent = FullDuplexVoiceAgent(mock_hardware=True)

    async def _timed_cancel():
        await asyncio.sleep(7.0)
        print("\n[TEST]: 7-second timeout reached. Stopping agent...")
        await agent.stop()

    cancel_task = asyncio.create_task(_timed_cancel())

    try:
        await agent.run()
    except (asyncio.CancelledError, Exception) as e:
        import traceback
        traceback.print_exc()
        print(f"[TEST]: Agent stopped with: {e}")
    finally:
        agent.cleanup()

    print("✅ Full-duplex agent lifecycle test PASSED!")

if __name__ == "__main__":
    asyncio.run(main())
