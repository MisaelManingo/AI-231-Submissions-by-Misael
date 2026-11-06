#!/usr/bin/env python3
"""
State Persistence for Alarms, Reminders, and Virtual Thermostat
- Atomic file writes (temp file -> replace) tolerant of power drops
- 24-hour auto-expiry for reminders
- Virtual Thermostat state persistence (mode & setpoint)
"""

import os
import json
import time
import threading
from config import STATE_FILE_PATH


class StateManager:
    """Manages atomic JSON persistence for assistant state."""
    def __init__(self, filepath=STATE_FILE_PATH):
        self.filepath = filepath
        self.lock = threading.Lock()
        self.state = {
            "reminders": [],  # List of {"task": str, "updated_at": float}
            "alarm": None,    # {"time": "6:00 AM", "hour": 6, "minute": 0, "active": True}
            "thermostat": {
                "setpoint": 22.0,
                "mode": "COOL",  # COOL, HEAT, AUTO, OFF
                "fan": "AUTO"
            }
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
                    if "thermostat" in data:
                        self.state["thermostat"].update(data["thermostat"])
            except Exception as e:
                print(f"[STATE WARNING]: Corrupt state file ({e}). Resetting to default.")
                self._save_unlocked()

        self._purge_expired_reminders()

    def _save_unlocked(self):
        tmp_path = str(self.filepath) + ".tmp"
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self.state, f, indent=2)
            os.replace(tmp_path, self.filepath)
        except Exception as e:
            print(f"[STATE ERROR]: Failed saving: {e}")

    def save(self):
        with self.lock:
            self._save_unlocked()

    def _purge_expired_reminders(self):
        """Purges reminders older than 24 hours (86,400s)."""
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

    def add_or_refresh_reminder(self, task_name: str) -> bool:
        """Adds reminder or refreshes 24h lifetime if exists. Returns already_exists."""
        self._purge_expired_reminders()
        task_clean = task_name.strip().lower()
        now = time.time()

        with self.lock:
            for item in self.state["reminders"]:
                if item["task"].lower() == task_clean:
                    item["updated_at"] = now
                    self._save_unlocked()
                    return True

            self.state["reminders"].append({
                "task": task_name.strip(),
                "updated_at": now
            })
            self._save_unlocked()
            return False

    def list_reminders(self) -> list:
        self._purge_expired_reminders()
        with self.lock:
            sorted_items = sorted(self.state["reminders"], key=lambda x: x.get("updated_at", 0))
            return [x["task"] for x in sorted_items]

    def clear_reminder(self, task_name: str) -> bool:
        with self.lock:
            orig = len(self.state["reminders"])
            self.state["reminders"] = [
                r for r in self.state["reminders"]
                if r["task"].lower() != task_name.strip().lower()
            ]
            changed = len(self.state["reminders"]) < orig
            if changed:
                self._save_unlocked()
            return changed

    def set_alarm(self, time_str: str, hour: int, minute: int):
        with self.lock:
            self.state["alarm"] = {
                "time": time_str,
                "hour": hour,
                "minute": minute,
                "active": True
            }
            self._save_unlocked()

    def get_alarm(self):
        with self.lock:
            return self.state.get("alarm")

    def deactivate_alarm(self):
        with self.lock:
            if self.state.get("alarm"):
                self.state["alarm"]["active"] = False
                self._save_unlocked()

    def get_thermostat(self) -> dict:
        with self.lock:
            return dict(self.state.get("thermostat", {}))

    def set_thermostat(self, setpoint: float = None, mode: str = None) -> dict:
        with self.lock:
            if setpoint is not None:
                self.state["thermostat"]["setpoint"] = float(setpoint)
            if mode is not None:
                self.state["thermostat"]["mode"] = mode.upper()
            self._save_unlocked()
            return dict(self.state["thermostat"])

