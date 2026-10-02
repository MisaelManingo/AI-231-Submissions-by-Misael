#!/usr/bin/env python3
"""
State Persistence, 24-Hour Reminders, Alarm & Timer Management
- Atomic JSON persistence (temp file -> replace) tolerant of corrupt/missing files
- 24-hour auto-expiry for reminders (oldest modified first)
- Duplicate reminder refresh logic
- Alarm manager matching RTC time against target alarm
- Active Buzzer (GPIO 6) and Push Button (GPIO 5) handlers
"""

import os
import json
import time
import threading
from datetime import datetime

from .config import (
    STATE_FILE_PATH, PIN_BUTTON, PIN_BUZZER, MOCK_HARDWARE
)

try:
    from gpiozero import Button, Buzzer
except (ImportError, Exception):
    Button = None
    Buzzer = None


class StateManager:
    """
    Manages persistent state for Alarms and Reminders in assistant_state.json.
    All disk writes are atomic to prevent file corruption.
    """
    def __init__(self, filepath=STATE_FILE_PATH):
        self.filepath = filepath
        self.lock = threading.Lock()
        self.state = {
            "reminders": [],  # List of {"task": str, "updated_at": float}
            "alarm": None     # {"time": "6:00 AM", "hour": 6, "minute": 0}
        }
        self.load()

    def load(self):
        with self.lock:
            if not os.path.exists(self.filepath):
                self._save_unlocked()
                return

            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.state["reminders"] = data.get("reminders", [])
                    self.state["alarm"] = data.get("alarm", None)
            except Exception as e:
                print(f"[STATE WARNING]: Corrupt state file ({e}). Initializing clean state.")
                self.state = {"reminders": [], "alarm": None}
                self._save_unlocked()

        self._purge_expired_reminders()

    def _save_unlocked(self):
        tmp_path = self.filepath + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
            os.replace(tmp_path, self.filepath)
        except Exception as e:
            print(f"[STATE SAVE ERROR]: Failed saving state: {e}")

    def save(self):
        with self.lock:
            self._save_unlocked()

    def _purge_expired_reminders(self):
        """Purges reminders older than 24 hours (86,400 seconds)."""
        now = time.time()
        cutoff = now - 86400.0
        with self.lock:
            orig_len = len(self.state["reminders"])
            self.state["reminders"] = [
                r for r in self.state["reminders"]
                if r.get("updated_at", 0) >= cutoff
            ]
            if len(self.state["reminders"]) != orig_len:
                self._save_unlocked()

    def add_or_refresh_reminder(self, task_name):
        """
        Adds reminder or refreshes 24h expiry if it already exists.
        Returns already_exists (bool).
        """
        self._purge_expired_reminders()
        task_clean = task_name.strip().lower()
        now = time.time()

        with self.lock:
            for item in self.state["reminders"]:
                if item["task"].lower() == task_clean:
                    item["updated_at"] = now
                    self._save_unlocked()
                    print(f"📋 [REMINDERS]: Refreshed 24h expiry for existing reminder: '{task_name}'")
                    return True

            self.state["reminders"].append({
                "task": task_name.strip(),
                "updated_at": now
            })
            self._save_unlocked()
            print(f"📋 [REMINDERS]: Added new reminder: '{task_name}'")
            return False

    def list_reminders(self):
        """Returns active reminders sorted by modification time (oldest first)."""
        self._purge_expired_reminders()
        with self.lock:
            sorted_items = sorted(self.state["reminders"], key=lambda x: x.get("updated_at", 0))
            return [x["task"] for x in sorted_items]

    def set_alarm(self, time_str, hour, minute):
        with self.lock:
            self.state["alarm"] = {
                "time": time_str,
                "hour": hour,
                "minute": minute,
                "active": True
            }
            self._save_unlocked()
            print(f"⏰ [ALARM]: Alarm set for {time_str}")

    def get_alarm(self):
        with self.lock:
            return self.state.get("alarm")


class AlarmAndTimerService:
    """
    Coordinates Alarm monitoring (RTC time) and Timer countdowns.
    Controls the physical Active Buzzer and Push Button.
    """
    def __init__(self, state_mgr, audio_engine, display_controller, rtc_sensor, mock=False):
        self.state_mgr = state_mgr
        self.audio = audio_engine
        self.display = display_controller
        self.rtc = rtc_sensor
        self.mock = mock or MOCK_HARDWARE or (Buzzer is None)

        self.buzzer = None
        self.button = None
        self.running = True

        self.timer_beeping = False
        self.alarm_firing = False
        self.pause_assistant_event = threading.Event()

        if not self.mock:
            try:
                self.buzzer = Buzzer(PIN_BUZZER)
                self.button = Button(PIN_BUTTON)
                self.button.when_pressed = self.on_button_pressed
                print(f"[ALARM/TIMER SERVICE]: Buzzer on GPIO {PIN_BUZZER}, Button on GPIO {PIN_BUTTON}")
            except Exception as e:
                print(f"[ALARM/TIMER WARNING]: GPIO initialization failed ({e}). Using mock.")
                self.mock = True

        self.worker = threading.Thread(target=self._alarm_monitor_loop, daemon=True)
        self.worker.start()

    def on_button_pressed(self):
        """Physical button stops buzzer beeping or active alarm."""
        print("🔘 [BUTTON PRESSED]: Silencing active alert!")
        if self.timer_beeping:
            self.stop_timer_beep()
        if self.alarm_firing:
            self.stop_alarm_fire()

    def start_timer(self, seconds):
        """Starts timer and registers completion callback."""
        self.stop_timer_beep()
        self.display.start_timer(seconds, on_finish_callback=self._on_timer_finished)

    def _on_timer_finished(self):
        """Called when timer expires. Starts beeping buzzer."""
        print("🔔 [TIMER EXPIRED]: Beeping active buzzer!")
        self.timer_beeping = True
        threading.Thread(target=self._beep_worker, daemon=True).start()

    def _beep_worker(self):
        """Beeps buzzer repeatedly for 30 seconds or until button is pressed."""
        start_t = time.time()
        while self.timer_beeping and (time.time() - start_t) < 30.0:
            if not self.mock and self.buzzer is not None:
                self.buzzer.on()
            time.sleep(0.3)
            if not self.mock and self.buzzer is not None:
                self.buzzer.off()
            time.sleep(0.3)

        self.stop_timer_beep()

    def stop_timer_beep(self):
        self.timer_beeping = False
        if not self.mock and self.buzzer is not None:
            self.buzzer.off()
        self.display.stop_timer()

    def _alarm_monitor_loop(self):
        """Monitors RTC clock against stored alarm."""
        last_checked_minute = -1

        while self.running:
            time.sleep(1.0)
            alarm_data = self.state_mgr.get_alarm()
            if not alarm_data or not alarm_data.get("active"):
                continue

            now_dt = self.rtc.read_time()
            if now_dt.minute == last_checked_minute:
                continue

            if now_dt.hour == alarm_data["hour"] and now_dt.minute == alarm_data["minute"]:
                last_checked_minute = now_dt.minute
                self._trigger_alarm_fire()

    def _trigger_alarm_fire(self):
        """
        Fires the alarm: STOP everything (wake listening, command listening, music, TTS),
        and play Alarm.mp3 in a loop (10 times) or until button is pressed.
        """
        print("🚨🚨🚨 [ALARM GOING OFF!]: Halting pipeline and playing Alarm.mp3...")
        self.alarm_firing = True
        self.pause_assistant_event.set()

        self.audio.stop_music()
        self.audio.trigger_alarm(repeats=10)

        def await_alarm():
            self.audio.alarm_event.wait()
            self.stop_alarm_fire()

        threading.Thread(target=await_alarm, daemon=True).start()

    def stop_alarm_fire(self):
        if self.alarm_firing:
            self.alarm_firing = False
            self.audio.stop_alarm()
            self.pause_assistant_event.clear()
            print("🚨 [ALARM RESUMED]: Voice Assistant pipeline resumed.")

    def close(self):
        self.running = False
        self.stop_timer_beep()
        self.stop_alarm_fire()
        if not self.mock:
            if self.buzzer is not None:
                try:
                    self.buzzer.close()
                except Exception:
                    pass
            if self.button is not None:
                try:
                    self.button.close()
                except Exception:
                    pass
