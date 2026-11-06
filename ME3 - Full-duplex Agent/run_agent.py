#!/usr/bin/env python3
"""
ME3 Full-Duplex Voice Assistant Entrypoint (Raspberry Pi 5)
Usage:
    source ~/AI_231_venv/bin/activate
    python run_agent.py [--mock] [--report]
"""

import sys
import asyncio
import signal
import argparse
from pathlib import Path

# Ensure project root in sys.path
BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from agent import FullDuplexVoiceAgent
from benchmark import generate_benchmark_report


def parse_args():
    parser = argparse.ArgumentParser(description="ME3 Full-Duplex Voice AI Agent on Raspberry Pi 5")
    parser.add_argument("--mock", action="store_true", help="Run with mock hardware (for simulation or laptop testing)")
    parser.add_argument("--report", action="store_true", default=True, help="Print benchmark summary report on shutdown")
    return parser.parse_args()


async def main_async(args):
    agent = FullDuplexVoiceAgent(mock_hardware=args.mock)

    loop = asyncio.get_running_loop()

    def handle_signal():
        print("\n[MAIN]: Caught interrupt signal. Initiating shutdown...")
        asyncio.create_task(agent.stop())

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, handle_signal)
        except NotImplementedError:
            pass

    try:
        await agent.run()
    except (asyncio.CancelledError, KeyboardInterrupt):
        pass
    finally:
        agent.cleanup()
        if args.report:
            print("\nGenerating turn latency benchmark report...")
            generate_benchmark_report()


def main():
    args = parse_args()
    try:
        asyncio.run(main_async(args))
    except KeyboardInterrupt:
        print("\n[MAIN]: Exited by user.")


if __name__ == "__main__":
    main()
