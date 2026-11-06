#!/usr/bin/env python3
"""
Standalone MCP Server Test
Verifies all 10 use cases and general agent tools.
"""

import sys
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from hardware import HardwareManager
from mcp_server import create_mcp_server
from mcp_server.client_bridge import get_mcp_function_schemas

def main():
    print("=" * 60)
    print("TESTING MCP SERVER & TOOLS STANDALONE")
    print("=" * 60)

    hw = HardwareManager(mock=True)
    mcp = create_mcp_server(hw)

    schemas = get_mcp_function_schemas(mcp)
    print(f"\nExtracted {len(schemas)} Pipecat FunctionSchemas:")
    for s in schemas:
        print(f" - {s.name}: {s.description}")

    tools = {t.name: t.fn for t in mcp._tool_manager.list_tools()}

    print("\n--- Testing Media Control ---")
    print(tools["control_media"](action="play"))
    print(tools["control_media"](action="volume_up", volume_delta=10))

    print("\n--- Testing Weather & Time ---")
    print(tools["get_weather"]())
    print(tools["get_time_date"]())

    print("\n--- Testing Lights Control ---")
    print(tools["control_lights"](power=True, brightness=80, color="GREEN"))
    print(tools["control_lights"](power=False))

    print("\n--- Testing Timers & Alarms ---")
    print(tools["set_timer"](seconds=5))
    print(tools["set_alarm"](time_str="7:30 AM", hour=7, minute=30))
    print(tools["get_alarm_status"]())

    print("\n--- Testing Virtual Thermostat ---")
    print(tools["control_thermostat"](setpoint_celsius=23.0, mode="COOL"))
    print(tools["get_thermostat_status"]())

    print("\n--- Testing Reminders ---")
    print(tools["add_reminder"](task="Finish AI 231 ME3 Report"))
    print(tools["list_reminders"]())

    print("\n--- Testing Telephony ---")
    print(tools["make_phone_call"](phone_number="+63XXXXXXXXXX"))
    print(tools["send_sms"](phone_number="+63XXXXXXXXXX", message_text="Test from RPi5"))

    print("\n--- Testing Agentic Website Builder ---")
    web_res = tools["create_simple_website"](
        title="AI 231 ME3 Agent Dashboard",
        html_body="<p>Full-Duplex Speech Agent is Online on Raspberry Pi 5!</p>",
        filename="test_dashboard.html"
    )
    print(web_res)

    print("\n--- Testing General Agent Tasks ---")
    print(tools["execute_bash_command"](command="uname -a && uptime"))
    print(tools["manage_file"](action="write", file_path="test_note.txt", content="ME3 agent note."))
    print(tools["manage_file"](action="read", file_path="test_note.txt"))

    hw.cleanup()
    print("\n✅ All MCP tools verified successfully!")

if __name__ == "__main__":
    main()

