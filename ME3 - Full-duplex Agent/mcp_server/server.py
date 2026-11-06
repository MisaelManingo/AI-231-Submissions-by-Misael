#!/usr/bin/env python3
"""
Model Context Protocol (MCP) Server for ME3 Full-Duplex Agent
Uses MCP 2.x MCPServer to declare and serve local tools for:
1. Media & Music Control
2. Live Weather (Open-Meteo keyless API) & System Time
3. Smart Lights (WS2813 NeoPixel strip on/off, brightness, colors)
4. Countdown Timers (TM1637 countdown + buzzer alert)
5. Alarms (Atomic JSON persistence + RTC/system time matching + buzzer)
6. Virtual Thermostat (software setpoint/mode + outdoor weather compare)
7. Reminders (24-hr auto-expiry atomic persistence)
8. Telephony (Android ADB voice call & SMS messaging)
9. Agentic Task: Website Builder (generates HTML & serves local HTTP preview)
10. General Agent Tasks: Terminal Command Execution & File Management
"""

import os
import re
import sys
import json
import time
import shutil
import urllib.request
import subprocess
import threading
import http.server
import socketserver
from datetime import datetime
from pathlib import Path

from mcp.server.mcpserver import MCPServer
from config import (
    WEATHER_LATITUDE, WEATHER_LONGITUDE, WEATHER_CITY_NAME,
    SERVED_PAGES_DIR, MUSIC_DIR, WORKSPACE_DIR, CALL_NUMBER, MESSAGE_NUMBER, CALL_CARRIER,
    GOOGLE_API_KEY, DELEGATE_AGENT_MODEL, DELEGATE_THINKING_LEVEL, DELEGATE_THINKING_BUDGET
)

# Global HTTP server state for website preview
_HTTP_SERVER = None
_HTTP_PORT = 8080
_BACKGROUND_TASKS = {}
_AGY_PATH_KEYS = ("path", "file_path", "directory_path", "TargetFile", "output_path", "script_path", "ScriptPath")


def _make_outside_workspace_check(ws_root: Path):
    """Predicate for Antigravity deny-policies: True if any path argument of a tool call leaves ws_root.
    (The SDK's own `workspaces` setting let an absolute /tmp write through in testing, so this is enforced explicitly.)"""
    ws = Path(ws_root).resolve()

    def outside(args: dict) -> bool:
        for key in _AGY_PATH_KEYS:
            val = (args or {}).get(key)
            if not isinstance(val, str) or not val:
                continue
            raw = val[7:] if val.startswith("file://") else val
            try:
                target = Path(raw).expanduser()
                if not target.is_absolute():
                    target = ws / target
                target.resolve().relative_to(ws)   # follows symlinks and ".." ; raises if outside
            except Exception:
                return True
        return False

    return outside


DELEGATE_TIMEOUT_S = int(os.getenv("DELEGATE_TIMEOUT_S", "900"))  # a stuck delegated task is abandoned after this
_DELEGATE_SYSTEM = (
    "You are an autonomous software engineer working for a voice assistant on a Raspberry Pi 5. "
    "SAFETY: your ONLY writable area is {ws}. Put everything you create inside {proj} (use absolute paths there). "
    "You cannot and must not touch any file outside {ws}, and you have no shell. "
    "QUALITY BAR: deliver COMPLETE, working, polished projects: no placeholders, no TODOs, no missing pieces. "
    "For games and web apps build a self-contained static site in the folder: index.html plus optional style.css/script.js, "
    "no build step, no external CDNs or network dependencies, runs offline in Chromium; games support keyboard AND mouse/touch, "
    "show score, game-over and restart. Use clear structure, requestAnimationFrame for animation, and handle edge cases. "
    "After writing, RE-READ your files and review the logic line by line (game loop, collisions, state reset, input handling, "
    "syntax) and fix any bug before finishing. Add a short README.md (how to open it, controls). "
    "FINAL ANSWER: at most two plain sentences, no markdown or code, because it is read aloud: say what you built and the "
    "exact path to open."
)


def get_weather_data(lat: float, lon: float):
    """Fetches real-time outdoor weather from Open-Meteo keyless REST API."""
    url = (
        f"https://api.open-meteo.com/v1/forecast?"
        f"latitude={lat}&longitude={lon}&current=temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m"
    )
    req = urllib.request.Request(url, headers={"User-Agent": "RPi5-AI231-Assistant"})
    with urllib.request.urlopen(req, timeout=5) as resp:
        data = json.loads(resp.read().decode())
    return data.get("current", {})


WEATHER_CODE_DESCRIPTIONS = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    95: "Thunderstorm",
}


def create_mcp_server(hardware_manager=None):
    """Creates and configures the MCP server with the hardware facade."""
    if hardware_manager is None:
        from hardware import HardwareManager
        hardware_manager = HardwareManager(mock=True)
    mcp = MCPServer("rpi-full-duplex-assistant")

    # -------------------------------------------------------------------------
    # 1. MEDIA & MUSIC CONTROL
    # -------------------------------------------------------------------------
    @mcp.tool()
    def control_media(action: str, volume_delta: int = 10, target_volume: int = 50) -> str:
        """
        Controls music and media playback from the media library.
        action: 'play', 'pause', 'stop', 'next', 'list_tracks', 'volume_up', 'volume_down', 'set_volume'
        """
        act = action.lower().strip()
        hardware_manager.on_thinking(f"Media: {act}")
        mp = getattr(hardware_manager, "music_player", None)

        if act == "play":
            if mp:
                title = mp.play()
                return f"Now playing music: '{title}' through the speakers."
            tracks = sorted([f.name for f in MUSIC_DIR.glob("*.mp3")]) if MUSIC_DIR.exists() else []
            track_name = tracks[0] if tracks else "Track"
            return f"Playing music: '{track_name}' from media folder."
        elif act == "pause":
            if mp:
                mp.pause()
            return "Music paused."
        elif act == "stop":
            if mp:
                mp.stop()
            return "Music stopped."
        elif act == "next":
            if mp:
                title = mp.next_track()
                return f"Skipped to next track: '{title}'."
            return "Skipped to next music track."
        elif act in ("list", "list_tracks", "tracks"):
            if mp:
                tracks = mp.get_track_list()
            else:
                tracks = sorted([f.name for f in MUSIC_DIR.glob("*.mp3")]) if MUSIC_DIR.exists() else []
            if not tracks:
                return "No music tracks found in media folder."
            return f"Available tracks in media folder: {', '.join(tracks)}"
        elif act == "volume_up":
            if mp:
                v = mp.adjust_volume(volume_delta)
                return f"Music volume increased to {v}%."
            return f"Volume increased by {volume_delta}%."
        elif act == "volume_down":
            if mp:
                v = mp.adjust_volume(-volume_delta)
                return f"Music volume decreased to {v}%."
            return f"Volume decreased by {volume_delta}%."
        elif act == "set_volume":
            if mp:
                v = mp.set_volume(target_volume)
                return f"Music volume set to {v}%."
            return f"Volume set to {target_volume}%."
        return f"Unknown media action: {action}"

    # -------------------------------------------------------------------------
    # 2. WEATHER & TIME/DATE
    # -------------------------------------------------------------------------
    @mcp.tool()
    def get_weather(city: str = WEATHER_CITY_NAME) -> str:
        """
        Gets current live outdoor weather from Open-Meteo REST API.
        Does NOT rely on internal DHT22 sensor.
        """
        hardware_manager.on_thinking("Fetching weather...")
        try:
            curr = get_weather_data(WEATHER_LATITUDE, WEATHER_LONGITUDE)
            temp_c = curr.get("temperature_2m", 25.0)
            humidity = curr.get("relative_humidity_2m", 60)
            wcode = curr.get("weather_code", 0)
            wind = curr.get("wind_speed_10m", 0.0)
            desc = WEATHER_CODE_DESCRIPTIONS.get(wcode, "Clear")
            return f"Current outdoor weather for {city}: {temp_c}°C, {desc}, {humidity}% humidity, wind {wind} km/h."
        except Exception as e:
            return f"Error querying weather service: {e}. Outdoor temperature is estimated at 26°C."

    @mcp.tool()
    def get_time_date() -> str:
        """
        Gets current local time and date from Raspberry Pi system clock.
        STRICT: ONLY invoke this when user explicitly asks 'what time is it', 'what is the date', or 'tell me the time'.
        Never invoke for general conversation or when user's intent is unclear.
        """
        now = datetime.now()
        time_str = now.strftime("%I:%M %p")
        date_str = now.strftime("%A, %B %d, %Y")
        return f"The current time is {time_str} on {date_str}."

    # -------------------------------------------------------------------------
    # 3. LIGHTS CONTROL (Grove WS2813 LED Strip)
    # -------------------------------------------------------------------------
    @mcp.tool()
    def control_lights(power: bool, brightness: int = 100, color: str = "RED") -> str:
        """
        Controls the 10-NeoPixel WS2813 LED strip.
        power: True to turn on, False to turn off.
        brightness: 0 to 100 percent.
        color: 'RED', 'GREEN', 'BLUE', 'WHITE', 'YELLOW', 'CYAN', 'PURPLE'
        """
        hardware_manager.on_thinking(f"Lights: {'ON' if power else 'OFF'}")
        hardware_manager.set_lights(power=power, brightness=brightness, color=color)
        if power:
            return f"Lights turned ON with {color} color at {brightness}% brightness."
        else:
            return "Lights turned OFF."

    # -------------------------------------------------------------------------
    # 4. TIMERS (TM1637 Display & Active Buzzer)
    # -------------------------------------------------------------------------
    @mcp.tool()
    def set_timer(seconds: int) -> str:
        """
        Starts a countdown timer for the specified duration in seconds.
        Displays countdown on the 4-digit TM1637 display and sounds active buzzer when complete.
        """
        hardware_manager.on_thinking(f"Timer: {seconds}s")
        hardware_manager.start_timer(seconds)
        return f"Timer set for {seconds} seconds. Countdown visible on the TM1637 display."

    @mcp.tool()
    def cancel_timer() -> str:
        """
        Cancels any running countdown timer and silences active buzzer.
        STRICT: ONLY invoke when user explicitly says 'cancel timer', 'stop timer', or 'reset timer'.
        Never invoke for unrelated speech, questions, or unclear user requests.
        """
        hardware_manager.alarm_timer.silence_alert()
        hardware_manager.tm1637.stop_timer()
        return "Active timer cancelled and buzzer silenced."

    # -------------------------------------------------------------------------
    # 5. ALARMS (Atomic JSON Persistence & System Clock Monitoring)
    # -------------------------------------------------------------------------
    @mcp.tool()
    def set_alarm(time_str: str, hour: int, minute: int) -> str:
        """
        Sets a persistent alarm.
        time_str: Human readable time (e.g. '6:00 AM', '8:30 PM')
        hour: Hour in 24-hour format (0 to 23)
        minute: Minute (0 to 59)
        """
        hardware_manager.on_thinking(f"Alarm: {time_str}")
        hardware_manager.set_alarm(time_str, hour, minute)
        return f"Alarm successfully set for {time_str}."

    @mcp.tool()
    def get_alarm_status() -> str:
        """Returns the currently active alarm, if any."""
        alarm = hardware_manager.state_mgr.get_alarm()
        if alarm and alarm.get("active"):
            return f"Active alarm is set for {alarm.get('time')}."
        return "No active alarm is currently set."

    @mcp.tool()
    def cancel_alarm() -> str:
        """
        Cancels the active scheduled alarm.
        STRICT: ONLY invoke when user explicitly says 'cancel alarm', 'stop alarm', or 'turn off alarm'.
        Never invoke for unrelated speech, questions, or unclear user requests.
        """
        hardware_manager.state_mgr.deactivate_alarm()
        hardware_manager.alarm_timer.silence_alert()
        return "Alarm deactivated."

    # -------------------------------------------------------------------------
    # 6. VIRTUAL THERMOSTAT (Software State & Outdoor Weather Compare)
    # -------------------------------------------------------------------------
    @mcp.tool()
    def control_thermostat(setpoint_celsius: float = None, mode: str = None) -> str:
        """
        Controls the virtual software thermostat.
        Note: The thermostat is a virtual software controller with NO indoor room sensor.
        It maintains software setpoint and mode, displays on LCD, and compares against outdoor weather.
        mode: 'COOL', 'HEAT', 'AUTO', 'OFF'
        setpoint_celsius: Target temperature in Celsius (e.g. 18.0, 22.0, 24.0)
        """
        hardware_manager.on_thinking("Thermostat...")
        state = hardware_manager.set_thermostat(setpoint=setpoint_celsius, mode=mode)
        try:
            curr = get_weather_data(WEATHER_LATITUDE, WEATHER_LONGITUDE)
            outdoor_temp = curr.get("temperature_2m", 25.0)
        except Exception:
            outdoor_temp = 25.0

        return (
            f"Virtual Thermostat set to {state['setpoint']}°C (Mode: {state['mode']}). "
            f"Current outdoor temperature is {outdoor_temp}°C."
        )

    @mcp.tool()
    def get_thermostat_status() -> str:
        """
        Gets current virtual thermostat state and compares with outdoor temperature.
        """
        state = hardware_manager.state_mgr.get_thermostat()
        try:
            curr = get_weather_data(WEATHER_LATITUDE, WEATHER_LONGITUDE)
            outdoor_temp = curr.get("temperature_2m", 25.0)
        except Exception:
            outdoor_temp = 25.0

        return (
            f"Virtual Thermostat setpoint: {state.get('setpoint')}°C, Mode: {state.get('mode')}. "
            f"Note: Virtual software thermostat has no indoor sensor; outdoor temperature is {outdoor_temp}°C."
        )

    # -------------------------------------------------------------------------
    # 7. REMINDERS (Atomic JSON Persistence with 24-Hour Expiry)
    # -------------------------------------------------------------------------
    @mcp.tool()
    def add_reminder(task: str) -> str:
        """
        Adds a persistent reminder with 24-hour auto-expiry.
        """
        hardware_manager.on_thinking(f"Reminder: {task}")
        already = hardware_manager.state_mgr.add_or_refresh_reminder(task)
        if already:
            return f"Reminder for '{task}' already existed; refreshed its 24-hour timer."
        return f"Reminder added: '{task}'."

    @mcp.tool()
    def list_reminders() -> str:
        """Lists all active persistent reminders (auto-purges older than 24h)."""
        rems = hardware_manager.state_mgr.list_reminders()
        if not rems:
            return "You have no active reminders."
        return "Your active reminders: " + "; ".join(rems) + "."

    @mcp.tool()
    def clear_reminder(task: str) -> str:
        """Clears a specific reminder by task description."""
        found = hardware_manager.state_mgr.clear_reminder(task)
        if found:
            return f"Cleared reminder: '{task}'."
        return f"No active reminder found matching '{task}'."

    # -------------------------------------------------------------------------
    # 8. TELEPHONY & MESSAGING (Android ADB / UIAutomator2)
    # -------------------------------------------------------------------------
    @mcp.tool()
    def make_phone_call(phone_number: str = CALL_NUMBER, carrier: str = CALL_CARRIER) -> str:
        """
        Initiates a voice phone call via connected Android smartphone.
        """
        hardware_manager.on_thinking(f"Calling {phone_number}")
        hardware_manager.telephony.make_call_async(phone_number, carrier)
        return f"Placing phone call to {phone_number} on carrier {carrier}."

    @mcp.tool()
    def send_sms(phone_number: str = MESSAGE_NUMBER, message_text: str = "Hello from Raspberry Pi") -> str:
        """
        Sends an SMS message via connected Android smartphone.
        """
        hardware_manager.on_thinking(f"SMS to {phone_number}")
        hardware_manager.telephony.send_sms_async(phone_number, message_text)
        return f"Sent SMS to {phone_number} with text: '{message_text}'."

    # -------------------------------------------------------------------------
    # 9. AGENTIC TASK: WEBSITE BUILDER
    # -------------------------------------------------------------------------
    @mcp.tool()
    def create_simple_website(title: str, html_body: str, filename: str = "index.html") -> str:
        """
        Generates a complete HTML website and hosts it on a local HTTP preview server.
        title: Title of webpage.
        html_body: Inner HTML body content.
        """
        hardware_manager.on_thinking("Building website...")
        global _HTTP_SERVER, _HTTP_PORT

        full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>{title}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
      margin: 0; padding: 2rem; background: #0f172a; color: #f8fafc;
    }}
    .container {{
      max-width: 800px; margin: 0 auto; background: #1e293b;
      padding: 2rem; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.3);
    }}
    h1 {{ color: #38bdf8; margin-top: 0; }}
    a {{ color: #38bdf8; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>{title}</h1>
    <div>{html_body}</div>
    <hr style="border-color: #334155; margin-top: 2rem;">
    <p style="font-size: 0.85rem; color: #94a3b8;">Created by ME3 Voice AI Assistant on Raspberry Pi 5</p>
  </div>
</body>
</html>
"""
        file_path = SERVED_PAGES_DIR / filename
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(full_html)

        # Start background preview web server if not already running
        if _HTTP_SERVER is None:
            def _serve():
                global _HTTP_SERVER
                handler = lambda *args: http.server.SimpleHTTPRequestHandler(*args, directory=str(SERVED_PAGES_DIR))
                try:
                    with socketserver.TCPServer(("", _HTTP_PORT), handler) as httpd:
                        _HTTP_SERVER = httpd
                        print(f"🌐 [HTTP SERVER]: Hosting {SERVED_PAGES_DIR} on port {_HTTP_PORT}")
                        httpd.serve_forever()
                except Exception as e:
                    print(f"[HTTP SERVER ERROR]: {e}")

            t = threading.Thread(target=_serve, daemon=True)
            t.start()

        return f"Website '{title}' created at {file_path}. Preview live at http://localhost:{_HTTP_PORT}/{filename}."

    # -------------------------------------------------------------------------
    # 10. GENERAL AGENT CAPABILITIES (Shell commands & File management)
    # -------------------------------------------------------------------------
    @mcp.tool()
    def execute_bash_command(command: str) -> str:
        """
        Executes a safe read-only shell command on the Raspberry Pi (e.g. 'uname -a', 'df -h', 'uptime', 'vcgencmd measure_temp').
        STRICT: ONLY invoke when user explicitly says 'run shell command' or 'execute bash'.
        NEVER invoke for media, music, lights, thermostat, timers, weather, or conversation.
        Destructive commands (reboot, shutdown, rm, poweroff, kill, mkfs, dd) are permanently blocked.
        """
        cmd_clean = command.strip()
        cmd_lower = cmd_clean.lower()

        # Strict safety guard: block any reboot, shutdown, poweroff, deletion, or destructive actions
        blocked_keywords = [
            "reboot", "shutdown", "poweroff", "halt", "init 0", "init 6",
            "rm -rf", "rm -r", "mkfs", "dd if=", "dd of=", ":(){ :|:& };:",
            "kill -9 -1", "killall", "systemctl reboot", "systemctl poweroff",
            "systemctl halt", "truncate", "wipefs", "> /dev/sd", "> /dev/mmc"
        ]
        if any(b in cmd_lower for b in blocked_keywords):
            print(f"🛑 [SECURITY BLOCKED]: Blocked dangerous bash command: '{command}'")
            return f"Execution blocked for safety: dangerous system command '{command}' is strictly forbidden on this assistant."

        hardware_manager.on_thinking(f"Run: {command[:15]}")
        try:
            res = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=10)
            out = res.stdout.strip()
            err = res.stderr.strip()
            if res.returncode == 0:
                return f"Output:\n{out[:400]}" if out else "Command executed successfully (no output)."
            else:
                return f"Command exited with code {res.returncode}. Error:\n{err[:400]}"
        except subprocess.TimeoutExpired:
            return "Command timed out after 10 seconds."
        except Exception as e:
            return f"Command execution failed: {e}"

    @mcp.tool()
    def manage_file(action: str, file_path: str = "", content: str = "") -> str:
        """
        Reads, writes, or checks files in the dedicated agent workspace/ folder.
        action: 'read', 'write', 'list'
        file_path: Relative path inside the workspace/ directory (e.g. 'notes.txt', 'scripts/test.py').
        """
        hardware_manager.on_thinking(f"File: {action}")
        WORKSPACE_DIR.mkdir(parents=True, exist_ok=True)
        
        # Resolve target path relative to dedicated workspace directory
        p = Path(file_path) if file_path else Path(".")
        if not p.is_absolute():
            p = (WORKSPACE_DIR / p).resolve()
        else:
            p = p.resolve()

        if action == "read":
            if not p.exists():
                return f"File '{p.name}' does not exist in workspace."
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()[:500]
        elif action == "write":
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "w", encoding="utf-8") as f:
                f.write(content)
            return f"Wrote {len(content)} characters to '{p.name}' in workspace/."
        elif action == "list":
            target_dir = p if p.is_dir() else WORKSPACE_DIR
            if not target_dir.is_dir():
                return f"Target directory does not exist."
            entries = [e.name for e in target_dir.iterdir()][:20]
            return f"Workspace directory contents: " + (", ".join(entries) if entries else "(empty)")
        return f"Unknown file action: {action}"

    # -------------------------------------------------------------------------
    # 11. BACKGROUND TASK DELEGATION (Antigravity Agent, Gemini 3.8 Flash Low/Medium Thinking)
    # -------------------------------------------------------------------------
    @mcp.tool()
    def delegate_long_task(task_name: str, task_instructions: str) -> str:
        """
        Delegates a long-running or complex agentic task to an asynchronous background
        worker running Antigravity Agent with Gemini 3.8 Flash (Low/Medium thinking).
        Returns immediately so full-duplex voice conversation is never blocked.
        USE THIS for anything that means writing code or building something: games ("flappy bird"), web apps and
        websites, scripts, programs, multi-file projects, reports. Never write that code yourself.
        task_name: short name (e.g. 'flappy bird').
        task_instructions: COMPLETE, detailed requirements (features, controls, look, behaviour, edge cases).
        The worker can only create/edit files inside the workspace/ folder, in a subfolder named after the task.
        """
        hardware_manager.on_thinking(f"Delegating: {task_name[:12]}")
        task_id = f"task_{len(_BACKGROUND_TASKS) + 1}_{int(time.time())}"
        _BACKGROUND_TASKS[task_id] = {
            "name": task_name,
            "status": "RUNNING",
            "started_at": time.time(),
            "result": None
        }

        # Every delegated task works in its own subfolder of workspace/ and nowhere else.
        slug = re.sub(r"[^a-z0-9]+", "_", task_name.lower()).strip("_")[:40] or "task"
        ws_root = Path(WORKSPACE_DIR).resolve()
        project_dir = ws_root / slug
        project_dir.mkdir(parents=True, exist_ok=True)
        _BACKGROUND_TASKS[task_id]["folder"] = str(project_dir)

        def _worker():
            res_text = None
            try:
                # 1. Primary: Autonomous Antigravity Agent with tool capabilities
                import asyncio
                from google.antigravity import (
                    Agent, LocalAgentConfig, CapabilitiesConfig,
                    ModelTarget, GeminiAPIEndpoint, GeminiModelOptions, ThinkingLevel
                )
                level = ThinkingLevel.MEDIUM if str(DELEGATE_THINKING_LEVEL).upper() == "MEDIUM" else ThinkingLevel.LOW
                model_name = DELEGATE_AGENT_MODEL.replace("models/", "")

                from google.antigravity import BuiltinTools
                from google.antigravity.hooks import policy as agy_policy
                from google.antigravity import RetryConfig, ModelAPIRetryConfig

                async def _run_antigravity():
                    config = LocalAgentConfig(
                        system_instructions=_DELEGATE_SYSTEM.format(ws=ws_root, proj=project_dir),
                        # No shell, no scheduling, no sub-agents: with only file tools left, "workspaces" below
                        # confines every read/write/create/edit to workspace/. Web search/URL read stay on so
                        # generated code can be checked against real documentation.
                        capabilities=CapabilitiesConfig(
                            enable_subagents=False,
                            disabled_tools=[BuiltinTools.RUN_COMMAND, BuiltinTools.SCHEDULE,
                                            BuiltinTools.INVOKE_SUBAGENT, BuiltinTools.RUN_WORKFLOW],
                        ),
                        workspaces=[str(ws_root)],
                        # Fail fast: by default the SDK silently retries 429/quota errors for minutes (looks like a hang).
                        retry_config=RetryConfig(api_retry=ModelAPIRetryConfig(max_retries=2, initial_sleep_duration_ms=2000)),
                        policies=[
                            *agy_policy.workspace_only([str(ws_root)]),
                            *[agy_policy.deny(t.value, when=_make_outside_workspace_check(ws_root),
                                              name="workspace_strict",
                                              reason=f"Only files inside {ws_root} may be touched.")
                              for t in (*BuiltinTools.file_tools(), BuiltinTools.LIST_DIR, BuiltinTools.SEARCH_DIR,
                                        BuiltinTools.FIND_FILE, BuiltinTools.GENERATE_IMAGE)],
                            *agy_policy.confirm_run_command(),
                        ],
                        model=ModelTarget(
                            name=model_name,
                            endpoint=GeminiAPIEndpoint(
                                api_key=GOOGLE_API_KEY,
                                options=GeminiModelOptions(thinking_level=level)
                            )
                        )
                    )
                    async def _chat():
                        async with Agent(config) as agent:
                            resp = await agent.chat(task_instructions)
                            tokens = []
                            async for token in resp:
                                tokens.append(token)
                            return "".join(tokens)
                    return await asyncio.wait_for(_chat(), timeout=DELEGATE_TIMEOUT_S)

                res_text = asyncio.run(_run_antigravity())
            except Exception as e_agy:
                _msg = str(e_agy)
                if "RESOURCE_EXHAUSTED" in _msg or "429" in _msg or "quota" in _msg.lower():
                    # The plain-Gemini fallback would hit the same quota, so report it clearly instead.
                    _BACKGROUND_TASKS[task_id]["status"] = (
                        f"FAILED: Gemini API quota exceeded for {DELEGATE_AGENT_MODEL} (the free tier allows only about "
                        f"20 requests per day). Try again later, or use a paid API key / another DELEGATE_AGENT_MODEL.")
                    print(f"❌ [BACKGROUND AGENT FAILED]: '{task_name}': API quota exceeded for {DELEGATE_AGENT_MODEL}", flush=True)
                    return
                print(f"⚠️ [ANTIGRAVITY NOTICE]: Antigravity agent encountered: {e_agy}. Falling back to standard Gemini Flash runner...")
                try:
                    from google import genai
                    from google.genai import types
                    client = genai.Client(api_key=GOOGLE_API_KEY)
                    resp = client.models.generate_content(
                        model=DELEGATE_AGENT_MODEL,
                        contents=task_instructions,
                        config=types.GenerateContentConfig(
                            thinking_config=types.ThinkingConfig(thinking_budget=DELEGATE_THINKING_BUDGET)
                        )
                    )
                    res_text = resp.text or "Task completed."
                    try:  # the plain-Gemini fallback cannot edit files itself: keep its answer in the task folder
                        (project_dir / "RESULT.md").write_text(res_text, encoding="utf-8")
                    except Exception:
                        pass
                except Exception as e_flash:
                    _BACKGROUND_TASKS[task_id]["status"] = f"FAILED: {e_flash}"
                    print(f"❌ [BACKGROUND AGENT FAILED]: '{task_name}': {e_flash}")
                    return

            if res_text:
                _BACKGROUND_TASKS[task_id]["status"] = "COMPLETED"
                _BACKGROUND_TASKS[task_id]["result"] = res_text
                print(f"🎉 [BACKGROUND AGENT COMPLETED]: '{task_name}'")
                hardware_manager.lcd.set_line(2, f"Done: {task_name[:14]}")
                hardware_manager.lcd.flush()
                hardware_manager.alarm_timer.buzzer.beep(duration=0.25)

        t = threading.Thread(target=_worker, daemon=True)
        t.start()

        return (
            f"Assigned '{task_name}' to background Antigravity agent running Gemini 3.8 Flash ({DELEGATE_THINKING_LEVEL} thinking). "
            f"Its files will be saved in workspace/{slug}/. I'll keep listening while it works, and signal you when it's finished."
        )

    @mcp.tool()
    def get_background_task_status(task_name: str = "") -> str:
        """Checks the status and output of background delegated tasks."""
        if not _BACKGROUND_TASKS:
            return "No background tasks have been delegated yet."
        reports = []
        for tid, tinfo in _BACKGROUND_TASKS.items():
            if not task_name or task_name.lower() in tinfo["name"].lower():
                reports.append(f"Task '{tinfo['name']}': {tinfo['status']} (files in {tinfo.get('folder', 'workspace/')})")
                if tinfo.get("result"):
                    reports.append(f"Output: {str(tinfo['result'])[:200]}")
        return "\n".join(reports) if reports else f"No task found matching '{task_name}'."

    return mcp

