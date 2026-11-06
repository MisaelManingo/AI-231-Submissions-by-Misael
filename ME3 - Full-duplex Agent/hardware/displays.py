#!/usr/bin/env python3
"""
Display Controllers for Raspberry Pi 5
1. 20x4 I2C Character LCD (PCF8574, address 0x27)
   Headless status display: LISTENING, THINKING, SPEAKING, BARGE_IN, ERROR.
   Double-buffered, flicker-free, throttled updates.
2. TM1637 4-Digit Display (CLK=GPIO 23, DIO=GPIO 24)
   Default: System clock HH:MM (blinking colon, internet-synced RPi system time)
   Countdown timer MM:SS
   Temperature popup XX°C
"""

import time
import threading
from datetime import datetime
from config import ADDR_LCD, PIN_TM1637_CLK, PIN_TM1637_DIO, MOCK_HARDWARE

try:
    from RPLCD.i2c import CharLCD
except (ImportError, Exception):
    CharLCD = None

try:
    import tm1637
except (ImportError, Exception):
    tm1637 = None


class LCDDisplay:
    """
    20x4 I2C Character LCD Driver.
    Serves as headless status display with double-buffering.
    """
    def __init__(self, addr=ADDR_LCD, mock=False):
        self.mock = mock or MOCK_HARDWARE or (CharLCD is None)
        self.lcd = None
        self.cols = 20
        self.rows = 4
        self.buffer = [" " * self.cols for _ in range(self.rows)]
        self.last_update_time = 0.0
        self.min_update_interval = 0.15  # ~6.6 Hz throttle to prevent I2C bus congestion
        self.lock = threading.Lock()
        self.current_state = "IDLE"
        # ASYNC WRITES: one full LCD refresh takes ~210 ms on this I2C bus. Done inline it froze the asyncio
        # event loop / audio path (300-400 ms frame gaps, distorted listen-probes). flush() now only snapshots
        # the screen; a daemon thread does the slow I2C writes. Revert: restore the inline loop in flush().
        self._pending = None
        self._wake = threading.Event()
        self._running = True
        self._writer_thread = None

        if not self.mock:
            try:
                self.lcd = CharLCD("PCF8574", addr, cols=self.cols, rows=self.rows)
                self.lcd.clear()
                print(f"[LCD DISPLAY]: 20x4 LCD connected at I2C address 0x{addr:02X}")
            except Exception as e:
                print(f"[LCD WARNING]: Could not open I2C LCD at 0x{addr:02X} ({e}). Using mock LCD.")
                self.mock = True

    def set_line(self, row: int, text: str):
        """Sets line in buffer, padded or truncated to 20 chars."""
        if 0 <= row < self.rows:
            formatted = f"{text:<20}"[:self.cols]
            self.buffer[row] = formatted

    def flush(self, force: bool = False):
        """Queues the buffer for the physical display (only if min interval passed or forced). Never blocks."""
        now = time.time()
        if not force and (now - self.last_update_time) < self.min_update_interval:
            return

        self.last_update_time = now
        if self.mock or self.lcd is None:
            return
        self._pending = list(self.buffer)  # snapshot; newest screen wins if the writer is busy
        if self._writer_thread is None:
            self._writer_thread = threading.Thread(target=self._writer, name="lcd-writer", daemon=True)
            self._writer_thread.start()
        self._wake.set()

    def _writer(self):
        while self._running:
            self._wake.wait()
            self._wake.clear()
            lines, self._pending = self._pending, None
            if lines is None or not self._running:
                continue
            with self.lock:
                try:
                    for row_idx, line in enumerate(lines):
                        self.lcd.cursor_pos = (row_idx, 0)
                        self.lcd.write_string(line)
                except Exception as e:
                    print(f"[LCD I2C ERROR]: Write failed: {e}")

    def update_vu(self, rms: float):
        """Updates line 1 with real-time audio VU meter (throttled)."""
        norm_rms = min(1.0, max(0.0, rms * 15.0))
        bars = int(norm_rms * 12)
        vu_bar = "#" * bars + "." * (12 - bars)
        self.set_line(1, f"VU: [{vu_bar}]")
        self.flush()

    def show_status(self, state: str, details: str = "", rms: float = 0.0):
        """
        Updates the 4-line headless status display:
        Row 0: State banner (e.g. [LISTENING], [THINKING], [SPEAKING])
        Row 1: Audio VU meter (RMS)
        Row 2: Action or context details
        Row 3: Clock & System status
        """
        self.current_state = state
        norm_rms = min(1.0, max(0.0, rms * 15.0))
        bars = int(norm_rms * 12)
        vu_bar = "#" * bars + "." * (12 - bars)

        now_str = datetime.now().strftime("%H:%M:%S")

        self.set_line(0, f"[{state.upper():<9}] {now_str}")
        self.set_line(1, f"VU: [{vu_bar}]")
        self.set_line(2, details[:20] if details else "Full-duplex Agent")
        self.set_line(3, "ME3 Pipecat + Gemini")
        self.flush()

    def clear(self):
        self.buffer = [" " * self.cols for _ in range(self.rows)]
        if not self.mock and self.lcd is not None:
            try:
                with self.lock:  # wait for an in-flight background write
                    self.lcd.clear()
            except Exception:
                pass

    def close(self):
        self._running = False
        self._wake.set()
        self.clear()
        if not self.mock and self.lcd is not None:
            try:
                self.lcd.backlight_enabled = False
            except Exception:
                pass


class TM1637Controller:
    """
    4-Digit 7-Segment Display Controller.
    Runs a background daemon thread managing:
    - Default: Current System Clock HH:MM with blinking colon (RPi internet time)
    - Countdown timer: Remaining MM:SS
    - Temperature popup: XX°C for duration
    """
    def __init__(self, clk_pin=PIN_TM1637_CLK, dio_pin=PIN_TM1637_DIO, mock=False):
        self.mock = mock or MOCK_HARDWARE or (tm1637 is None)
        self.tm = None
        self.lock = threading.Lock()
        self.running = True

        # Modes: 'CLOCK', 'TEMP_POPUP', 'TIMER'
        self.mode = "CLOCK"
        self.temp_val = 24
        self.temp_expiry = 0.0

        self.timer_seconds_left = 0
        self.timer_end_time = 0.0
        self.timer_callback = None

        if not self.mock:
            try:
                self.tm = tm1637.TM1637(clk=clk_pin, dio=dio_pin)
                self.tm.brightness(val=3)
                print(f"[TM1637]: Initialized on CLK={clk_pin}, DIO={dio_pin}")
            except Exception as e:
                print(f"[TM1637 WARNING]: Failed initializing TM1637 ({e}). Using mock.")
                self.mock = True

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def show_temperature_popup(self, temp_c: float, duration: float = 5.0):
        """Displays temperature popup on 4-digit display."""
        with self.lock:
            self.temp_val = int(round(temp_c))
            self.temp_expiry = time.time() + duration
            self.mode = "TEMP_POPUP"

    def start_timer(self, seconds: int, on_finish=None):
        """Starts background countdown timer."""
        with self.lock:
            self.timer_seconds_left = seconds
            self.timer_end_time = time.time() + seconds
            self.timer_callback = on_finish
            self.mode = "TIMER"

    def stop_timer(self):
        with self.lock:
            self.mode = "CLOCK"
            self.timer_callback = None

    def _worker_loop(self):
        colon_state = True
        while self.running:
            try:
                now = time.time()
                with self.lock:
                    current_mode = self.mode

                    # Check temp popup expiration
                    if current_mode == "TEMP_POPUP" and now > self.temp_expiry:
                        current_mode = "CLOCK"
                        self.mode = "CLOCK"

                    # Check timer countdown
                    if current_mode == "TIMER":
                        remaining = int(self.timer_end_time - now)
                        if remaining <= 0:
                            current_mode = "CLOCK"
                            self.mode = "CLOCK"
                            cb = self.timer_callback
                            self.timer_callback = None
                            if cb:
                                threading.Thread(target=cb, daemon=True).start()
                        else:
                            self.timer_seconds_left = remaining

                # Render display
                if not self.mock and self.tm is not None:
                    if current_mode == "TEMP_POPUP":
                        # Format "XX*C" -> e.g. [2, 4, degree, C]
                        self.tm.temperature(self.temp_val)
                    elif current_mode == "TIMER":
                        mins = self.timer_seconds_left // 60
                        secs = self.timer_seconds_left % 60
                        self.tm.numbers(mins, secs, colon=True)
                    else:
                        # Clock from RPi internet system time
                        dt = datetime.now()
                        self.tm.numbers(dt.hour, dt.minute, colon=colon_state)

                colon_state = not colon_state
                time.sleep(0.5)
            except Exception as e:
                time.sleep(1.0)

    def close(self):
        self.running = False
        if not self.mock and self.tm is not None:
            try:
                self.tm.write([0, 0, 0, 0])
            except Exception:
                pass

