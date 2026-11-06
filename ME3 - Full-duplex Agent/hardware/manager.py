#!/usr/bin/env python3
"""
Hardware Manager Facade for ME3 Full-Duplex Assistant
Coordinates all peripherals:
- 20x4 I2C LCD (Headless status: LISTENING, THINKING, SPEAKING, ERROR)
- TM1637 4-digit display (RPi internet clock, temp popup, timer)
- Status RGB LED (Yellow, Blue, Green, Red)
- Grove NeoPixel LED Strip (10x WS2813 on SPI MOSI)
- Active Buzzer & Push Button
- State Manager (Alarms, Reminders, Virtual Thermostat)
- Telephony (Call & SMS)
"""

import time
from config import MOCK_HARDWARE
from .displays import LCDDisplay, TM1637Controller
from .leds import StatusRGBLED, GroveLEDStrip
from .alerts import AlarmAndTimerService
from .state_manager import StateManager
from .telephony import TelephonyController


class HardwareManager:
    """Facade providing high-level interface to all hardware peripherals."""
    def __init__(self, mock=False):
        self.mock = mock or MOCK_HARDWARE
        print("\n" + "=" * 60)
        print("INITIALIZING RASPBERRY PI 5 HARDWARE SUBSYSTEMS")
        print(f"Mode: {'MOCK (Simulation)' if self.mock else 'PHYSICAL HARDWARE (3.3V GPIO/I2C/SPI)'}")
        print("=" * 60)

        # 1. State Persistence
        self.state_mgr = StateManager()

        # 2. Displays
        self.lcd = LCDDisplay(mock=self.mock)
        self.tm1637 = TM1637Controller(mock=self.mock)

        # 3. LEDs
        self.status_led = StatusRGBLED(mock=self.mock)
        self.strip = GroveLEDStrip(mock=self.mock)

        # 4. Alerts (Buzzer & Button)
        self.alarm_timer = AlarmAndTimerService(
            state_mgr=self.state_mgr,
            display_controller=self.tm1637,
            mock=self.mock
        )

        # 5. Telephony
        self.telephony = TelephonyController(mock=self.mock)

        # 6. Music Player (injected by AudioSystem)
        self.music_player = None

        # 7. GUI State Publisher (optional hook)
        self.publisher = None

        # Initialize to Idle State
        self.on_idle()
        print("[HARDWARE MANAGER]: All subsystems online.\n")

    # -------------------------------------------------------------------------
    # HEADLESS STATUS UI HOOKS (Triggered by Full-Duplex Agent Loop)
    # -------------------------------------------------------------------------
    def update_vu(self, rms: float):
        """Updates live VU meter on LCD row 1."""
        self.lcd.update_vu(rms)

    def on_idle(self):
        """Assistant is standing by and listening for speech."""
        self.status_led.set_yellow()
        self.lcd.show_status("LISTENING", "Ready for speech", rms=0.0)
        if getattr(self, "publisher", None):
            self.publisher.set_state("idle")

    def on_listening(self):
        """Alias for on_idle()."""
        self.on_idle()

    def on_user_speaking(self, rms: float = 0.0):
        """User is actively talking."""
        self.status_led.set_listening()  # was set_yellow(); distinct listening colour to match GUI
        self.lcd.show_status("LISTENING", "User speaking...", rms=rms)
        if getattr(self, "publisher", None):
            self.publisher.set_state("listening", rms=rms)

    def on_thinking(self, action_name: str = "Processing..."):
        """Model or tool execution is in progress."""
        self.status_led.set_blue()
        self.lcd.show_status("THINKING", action_name, rms=0.0)
        if getattr(self, "publisher", None):
            self.publisher.set_state("thinking")

    def on_speaking(self, rms: float = 0.0):
        """Agent is actively streaming audio response on speaker."""
        self.status_led.set_green()
        self.lcd.show_status("SPEAKING", "Voice output active", rms=rms)
        if getattr(self, "publisher", None):
            self.publisher.set_state("speaking", rms=rms)

    def on_barge_in(self):
        """User interrupted model playback."""
        self.status_led.set_yellow()
        self.lcd.show_status("BARGE_IN", "Interrupted!", rms=0.1)
        if getattr(self, "publisher", None):
            self.publisher.set_state("idle")

    def on_error(self, message: str = "Error occurred"):
        """System or API error."""
        self.status_led.set_red()
        self.lcd.show_status("ERROR", message[:20], rms=0.0)
        if getattr(self, "publisher", None):
            self.publisher.set_state("idle")

    # -------------------------------------------------------------------------
    # SMART-HOME PHYSICAL ACTIONS
    # -------------------------------------------------------------------------
    def set_lights(self, power: bool, brightness: int = 100, color: str = "RED"):
        if power:
            self.strip.set_color(color)
            self.strip.set_brightness(brightness)
            self.strip.turn_on()
        else:
            self.strip.turn_off()

    def set_thermostat(self, setpoint: float = None, mode: str = None) -> dict:
        state = self.state_mgr.set_thermostat(setpoint, mode)
        if setpoint is not None:
            self.tm1637.show_temperature_popup(setpoint, duration=5.0)
        self.lcd.set_line(2, f"Thermo: {state['setpoint']}C {state['mode']}")
        self.lcd.flush()
        return state

    def start_timer(self, seconds: int):
        self.alarm_timer.start_timer(seconds)

    def set_alarm(self, time_str: str, hour: int, minute: int):
        self.state_mgr.set_alarm(time_str, hour, minute)

    def cleanup(self):
        print("\n[HARDWARE MANAGER]: Cleaning up GPIO and displays...")
        try:
            self.status_led.close()
        except Exception:
            pass
        try:
            self.strip.close()
        except Exception:
            pass
        try:
            self.lcd.close()
        except Exception:
            pass
        try:
            self.tm1637.close()
        except Exception:
            pass
        try:
            self.alarm_timer.close()
        except Exception:
            pass
        print("[HARDWARE MANAGER]: Cleanup complete.")

