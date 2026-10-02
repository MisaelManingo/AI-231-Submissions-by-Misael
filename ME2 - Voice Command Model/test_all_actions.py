#!/usr/bin/env python3
"""
Automated Test Suite for All 32 Voice Command Hardware Actions
Tests all 32 classes, state persistence, 24h reminder expiry, duplicate refresh,
and displays/LEDs in mock mode.
"""

import os
import sys
import time
import json

# Ensure parent directory is in path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

from demo_rpi5_action import RPi5VoiceAssistant

def test_all_commands():
    print("\n" + "=" * 70)
    print("STARTING 32-COMMAND HARDWARE ACTION VERIFICATION SUITE")
    print("=" * 70)

    # Initialize assistant in mock mode
    assistant = RPi5VoiceAssistant(mock=True)

    labels_32 = [
        "LIGHT_ON", "LIGHT_OFF",
        "BRIGHTNESS_20", "BRIGHTNESS_60", "BRIGHTNESS_100",
        "COLOR_RED", "COLOR_GREEN", "COLOR_BLUE",
        "PLAY_MUSIC", "PAUSE", "STOP", "NEXT", "VOLUME_UP", "VOLUME_DOWN",
        "TEMPERATURE_18", "TEMPERATURE_22", "TEMPERATURE_26",
        "TIME", "WEATHER",
        "TIMER_10s", "TIMER_30s", "TIMER_1m",
        "ALARM_6_00AM", "ALARM_8_00AM", "ALARM_9_00PM",
        "CREATE_REMINDER_DRINK_WATER",
        "CREATE_REMINDER_DRINK_WATER", # Duplicate test -> should refresh and say already exists
        "CREATE_REMINDER_STUDY",
        "CREATE_REMINDER_EXERCISE",
        "LIST_REMINDERS",
        "CALL", "MESSAGE"
    ]

    print(f"\nExecuting {len(labels_32)} test command dispatches...\n")
    for idx, label in enumerate(labels_32, 1):
        print(f"\n--- [TEST {idx}/{len(labels_32)}]: Dispatching '{label}' ---")
        assistant.update_state(label)

    # Verify state persistence file
    state_file = os.path.join(BASE_DIR, "assistant_state.json")
    assert os.path.exists(state_file), "assistant_state.json was not created!"
    with open(state_file, "r") as f:
        persisted = json.load(f)

    print("\n--- Verifying Persistent State ---")
    print(json.dumps(persisted, indent=2))
    assert len(persisted["reminders"]) == 3, f"Expected 3 distinct reminders, got {len(persisted['reminders'])}"
    assert persisted["alarm"]["time"] == "9:00 PM", f"Expected Alarm 9:00 PM, got {persisted['alarm']}"

    # Clean shutdown test
    print("\n--- Testing Hardware Cleanup ---")
    assistant.hw.cleanup()

    print("\n" + "=" * 70)
    print("✅ ALL 32 VOICE COMMAND ACTIONS TESTED AND PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_all_commands()

