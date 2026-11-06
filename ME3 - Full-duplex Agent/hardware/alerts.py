#!/usr/bin/env python3
"""
Buzzer and Button Alert Controllers for Raspberry Pi 5
- Active Buzzer (GPIO 6)
- Push Button (GPIO 5, internal pull-up, active low)
- Alarm & Timer alerting with button silence
"""

import time
import threading
from datetime import datetime
from config import PIN_BUZZER, PIN_BUTTON, MOCK_HARDWARE

try:
    from gpiozero import Buzzer, Button
except (ImportError, Exception):
    Buzzer = None
    Button = None


class ActiveBuzzer:
    """Controls the active buzzer on GPIO 6."""
    def __init__(self, pin=PIN_BUZZER, mock=False):
        self.mock = mock or MOCK_HARDWARE or (Buzzer is None)
        self.buzzer = None
        self.beeping = False

        if not self.mock:
            try:
                self.buzzer = Buzzer(pin)
                print(f"[BUZZER]: Initialized on GPIO {pin}")
            except Exception as e:
                print(f"[BUZZER WARNING]: Failed initializing buzzer ({e}). Using mock.")
                self.mock = True

    def beep(self, duration: float = 0.2):
        """Short single beep."""
        if not self.mock and self.buzzer is not None:
            try:
                self.buzzer.on()
                time.sleep(duration)
                self.buzzer.off()
            except Exception:
                pass
        else:
            print(f"🔊 [MOCK BUZZER]: BEEP ({duration:.2f}s)")

    def start_alarm_pattern(self, stop_event: threading.Event):
        """Beeps periodically until stop_event is set."""
        while not stop_event.is_set():
            if not self.mock and self.buzzer is not None:
                try:
                    self.buzzer.on()
                    time.sleep(0.15)
                    self.buzzer.off()
                    time.sleep(0.15)
                    self.buzzer.on()
                    time.sleep(0.15)
                    self.buzzer.off()
                    time.sleep(0.6)
                except Exception:
                    break
            else:
                print("🔊 [MOCK BUZZER]: ALARM BEEP-BEEP...")
                time.sleep(1.0)

    def close(self):
        if not self.mock and self.buzzer is not None:
            try:
                self.buzzer.off()
                self.buzzer.close()
            except Exception:
                pass


class PushButtonListener:
    """Listens for push button press on GPIO 5."""
    def __init__(self, pin=PIN_BUTTON, on_press=None, mock=False):
        self.mock = mock or MOCK_HARDWARE or (Button is None)
        self.button = None
        self.on_press_callback = on_press

        if not self.mock:
            try:
                self.button = Button(pin, pull_up=True, bounce_time=0.05)
                if self.on_press_callback:
                    self.button.when_pressed = self._handle_press
                print(f"[BUTTON]: Initialized on GPIO {pin}")
            except Exception as e:
                print(f"[BUTTON WARNING]: Failed initializing button ({e}). Using mock.")
                self.mock = True

    def _handle_press(self):
        print("🔘 [BUTTON]: Pressed!")
        if self.on_press_callback:
            self.on_press_callback()

    def set_callback(self, cb):
        self.on_press_callback = cb
        if not self.mock and self.button is not None:
            self.button.when_pressed = self._handle_press

    def close(self):
        if not self.mock and self.button is not None:
            try:
                self.button.close()
            except Exception:
                pass


class AlarmAndTimerService:
    """
    Coordinates Alarms (checked against system clock) and Timers,
    triggering buzzer alert loops until silenced by push button or API.
    """
    def __init__(self, state_mgr, display_controller=None, mock=False):
        self.state_mgr = state_mgr
        self.display = display_controller
        self.mock = mock

        self.buzzer = ActiveBuzzer(mock=mock)
        self.button = PushButtonListener(on_press=self.silence_alert, mock=mock)

        self.running = True
        self.active_alert_event = None
        self.last_alarm_minute_triggered = None

        self.worker_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.worker_thread.start()

    def silence_alert(self):
        """Silences any actively sounding buzzer."""
        if self.active_alert_event:
            print("🔕 [ALERTS]: Alert silenced.")
            self.active_alert_event.set()
            self.active_alert_event = None

    def trigger_alert(self, title: str):
        """Starts alarm sound loop in background."""
        print(f"🚨 [ALERT TRIGGERED]: {title}")
        self.silence_alert()
        stop_event = threading.Event()
        self.active_alert_event = stop_event
        t = threading.Thread(target=self.buzzer.start_alarm_pattern, args=(stop_event,), daemon=True)
        t.start()

    def start_timer(self, seconds: int):
        """Starts timer and alerts when finished."""
        if self.display:
            self.display.start_timer(seconds, on_finish=lambda: self.trigger_alert(f"Timer ({seconds}s) Expired!"))
        else:
            threading.Timer(seconds, lambda: self.trigger_alert("Timer Expired!")).start()

    def _monitor_loop(self):
        """Checks for alarms against system time every 2 seconds."""
        while self.running:
            try:
                now = datetime.now()
                alarm = self.state_mgr.get_alarm()
                if alarm and alarm.get("active", False):
                    target_hr = alarm.get("hour")
                    target_min = alarm.get("minute")
                    cur_key = (now.hour, now.minute)
                    if cur_key == (target_hr, target_min) and self.last_alarm_minute_triggered != cur_key:
                        self.last_alarm_minute_triggered = cur_key
                        self.trigger_alert(f"Alarm for {alarm.get('time', 'now')}!")
                        # Disable one-time alarm
                        self.state_mgr.deactivate_alarm()
            except Exception:
                pass
            time.sleep(2)

    def close(self):
        self.running = False
        self.silence_alert()
        self.buzzer.close()
        self.button.close()

