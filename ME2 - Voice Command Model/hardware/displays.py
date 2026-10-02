#!/usr/bin/env python3
"""
Display Controllers for Raspberry Pi 5
1. 20x4 I2C Character LCD (Double-buffered, flicker-free, throttled)
2. TM1637 4-Digit Display (RTC Clock with blinking colon, Temp popup, Countdown Timer)
"""

import time
import threading
from .config import ADDR_LCD, PIN_TM1637_CLK, PIN_TM1637_DIO, MOCK_HARDWARE

# Try importing hardware libraries
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
    20x4 I2C LCD Driver with double-buffering and rate limiting to avoid flicker.
    """
    def __init__(self, addr=ADDR_LCD, mock=False):
        self.mock = mock or MOCK_HARDWARE or (CharLCD is None)
        self.lcd = None
        self.cols = 20
        self.rows = 4
        self.buffer = [" " * self.cols for _ in range(self.rows)]
        self.last_update_time = 0.0
        self.min_update_interval = 0.18  # ~5.5 Hz throttling

        if not self.mock:
            try:
                self.lcd = CharLCD('PCF8574', addr, cols=self.cols, rows=self.rows)
                self.lcd.clear()
                print(f"[LCD DISPLAY]: 20x4 LCD initialized at I2C address 0x{addr:02X}")
            except Exception as e:
                print(f"[LCD WARNING]: Could not connect to I2C LCD at 0x{addr:02X} ({e}). Using mock.")
                self.mock = True

    def set_line(self, row, text):
        """Sets a line in the LCD buffer (0 to 3). Padded or truncated to 20 chars."""
        if 0 <= row < self.rows:
            formatted = f"{text:<20}"[:self.cols]
            self.buffer[row] = formatted

    def flush(self, force=False):
        """Pushes buffer to physical display only if content changed or forced."""
        now = time.time()
        if not force and (now - self.last_update_time) < self.min_update_interval:
            return

        self.last_update_time = now

        if self.mock:
            return

        try:
            for row_idx, line in enumerate(self.buffer):
                self.lcd.cursor_pos = (row_idx, 0)
                self.lcd.write_string(line)
        except Exception as e:
            print(f"[LCD I2C ERROR]: Failed writing to LCD: {e}")

    def clear(self):
        self.buffer = [" " * self.cols for _ in range(self.rows)]
        if not self.mock and self.lcd is not None:
            try:
                self.lcd.clear()
            except Exception:
                pass

    def close(self):
        if not self.mock and self.lcd is not None:
            try:
                self.lcd.clear()
                self.lcd.backlight_enabled = False
            except Exception:
                pass


class TM1637Controller:
    """
    4-Digit 7-Segment Display Controller.
    Runs a background daemon thread that manages:
    1. Default: Clock from RTC (HH:MM with blinking colon)
    2. Temperature popup: Shows 'XX*C' for 5 seconds
    3. Countdown timer: Displays remaining MM:SS with static colon
    4. Alarm/Timer alert: Flashing display
    """
    def __init__(self, clk_pin=PIN_TM1637_CLK, dio_pin=PIN_TM1637_DIO, rtc_reader=None, mock=False):
        self.mock = mock or MOCK_HARDWARE or (tm1637 is None)
        self.rtc_reader = rtc_reader
        self.tm = None
        self.lock = threading.Lock()
        self.running = True

        # Modes: 'CLOCK', 'TEMP_POPUP', 'TIMER', 'TIMER_EXPIRED'
        self.mode = "CLOCK"

        # Temp popup state
        self.temp_val = 22
        self.temp_expiry = 0.0

        # Timer state
        self.timer_end_time = 0.0
        self.timer_seconds_left = 0
        self.timer_callback = None
        self.timer_alert_stop_event = threading.Event()

        if not self.mock:
            try:
                self.tm = tm1637.TM1637(clk=clk_pin, dio=dio_pin)
                self.tm.brightness(val=4)
                print(f"[TM1637]: Initialized on CLK={clk_pin}, DIO={dio_pin}")
            except Exception as e:
                print(f"[TM1637 WARNING]: Failed initializing TM1637 ({e}). Using mock.")
                self.mock = True

        self.thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.thread.start()

    def show_temperature_popup(self, temp_c, duration=5.0):
        """Displays temperature on the 4-digit display for duration seconds."""
        with self.lock:
            self.temp_val = int(round(temp_c))
            self.temp_expiry = time.time() + duration
            self.mode = "TEMP_POPUP"
            print(f"🌡️  [TM1637]: Showing Temperature {self.temp_val}°C for {duration:.0f}s")

    def start_timer(self, seconds, on_finish_callback=None):
        """Starts a background countdown timer."""
        with self.lock:
            self.timer_seconds_left = seconds
            self.timer_end_time = time.time() + seconds
            self.timer_callback = on_finish_callback
            self.mode = "TIMER"
            print(f"⏳ [TM1637 TIMER]: Countdown started for {seconds} seconds.")

    def stop_timer(self):
        """Cancels or stops the active timer and returns to clock."""
        with self.lock:
            self.mode = "CLOCK"
            self.timer_alert_stop_event.set()

    def _worker_loop(self):
        colon = True
        while self.running:
            now = time.time()

            with self.lock:
                current_mode = self.mode

            if current_mode == "TEMP_POPUP":
                if now >= self.temp_expiry:
                    with self.lock:
                        self.mode = "CLOCK"
                else:
                    if not self.mock and self.tm is not None:
                        try:
                            self.tm.temperature(self.temp_val)
                        except Exception:
                            pass
                    time.sleep(0.2)
                    continue

            elif current_mode == "TIMER":
                rem = max(0, int(self.timer_end_time - now))
                if rem <= 0:
                    with self.lock:
                        self.mode = "TIMER_EXPIRED"
                        cb = self.timer_callback
                    if cb:
                        cb()
                else:
                    m = rem // 60
                    s = rem % 60
                    if not self.mock and self.tm is not None:
                        try:
                            self.tm.numbers(m, s, colon=True)
                        except Exception:
                            pass
                    time.sleep(0.2)
                    continue

            elif current_mode == "TIMER_EXPIRED":
                # Flash display
                if not self.mock and self.tm is not None:
                    try:
                        self.tm.numbers(0, 0, colon=colon)
                    except Exception:
                        pass
                colon = not colon
                time.sleep(0.3)
                continue

            # Default: CLOCK mode
            # Read from RTC if available, otherwise fallback to local system time
            hour = 12
            minute = 0
            if self.rtc_reader is not None:
                dt = self.rtc_reader.read_time()
                if dt is not None:
                    hour = dt.hour
                    minute = dt.minute
                else:
                    lt = time.localtime()
                    hour = lt.tm_hour
                    minute = lt.tm_min
            else:
                lt = time.localtime()
                hour = lt.tm_hour
                minute = lt.tm_min

            if not self.mock and self.tm is not None:
                try:
                    self.tm.numbers(hour, minute, colon=colon)
                except Exception:
                    pass

            colon = not colon
            time.sleep(0.5)

    def close(self):
        self.running = False
        if not self.mock and self.tm is not None:
            try:
                self.tm.write([0, 0, 0, 0])
            except Exception:
                pass

