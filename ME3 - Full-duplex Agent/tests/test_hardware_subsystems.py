#!/usr/bin/env python3
"""
Test Hardware Subsystems
Verifies mock and physical hardware operation.
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import time
import argparse
from hardware import HardwareManager

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mock", action="store_true", help="Run with mock hardware")
    args = parser.parse_args()

    hw = HardwareManager(mock=args.mock)
    print("Testing status states...")
    hw.on_idle()
    time.sleep(1)

    hw.on_user_speaking(rms=0.05)
    time.sleep(1)

    hw.on_thinking("MCP Tool: Lights")
    time.sleep(1)

    hw.on_speaking(rms=0.08)
    time.sleep(1)

    print("Testing lights...")
    hw.set_lights(power=True, brightness=60, color="BLUE")
    time.sleep(1)
    hw.set_lights(power=False)

    print("Testing virtual thermostat...")
    res = hw.set_thermostat(setpoint=21.5, mode="COOL")
    print("Thermostat state:", res)
    time.sleep(1)

    print("Testing buzzer single beep...")
    hw.alarm_timer.buzzer.beep(0.1)

    print("Cleaning up...")
    hw.cleanup()
    print("✅ Hardware subsystem test PASSED!")

if __name__ == "__main__":
    main()
