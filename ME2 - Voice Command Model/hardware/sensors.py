#!/usr/bin/env python3
"""
Sensor Drivers for Raspberry Pi 5
1. DS3231 Real-Time Clock (I2C 0x68, Read-Only)
2. DHT22 Temperature & Humidity Sensor (Adafruit DHT on GPIO 4)
"""

import time
from datetime import datetime
from .config import ADDR_RTC, I2C_BUS, PIN_DHT22_DATA, MOCK_HARDWARE

try:
    from smbus2 import SMBus
except (ImportError, Exception):
    SMBus = None

try:
    import board
    import adafruit_dht
except (ImportError, Exception):
    board = None
    adafruit_dht = None


def bcd2dec(b):
    return (b >> 4) * 10 + (b & 0x0F)


class DS3231RTC:
    """
    DS3231 RTC Driver via SMBus2.
    Strictly READ-ONLY as required: never calls set_time.
    """
    def __init__(self, bus_num=I2C_BUS, addr=ADDR_RTC, mock=False):
        self.mock = mock or MOCK_HARDWARE or (SMBus is None)
        self.bus_num = bus_num
        self.addr = addr
        self.bus = None

        if not self.mock:
            try:
                self.bus = SMBus(bus_num)
                # Test read status byte
                self.bus.read_byte_data(self.addr, 0x0F)
                print(f"[RTC DS3231]: Connected on I2C bus {bus_num}, address 0x{addr:02X}")
            except Exception as e:
                print(f"[RTC WARNING]: Failed accessing DS3231 at 0x{addr:02X} ({e}). Using mock/system clock.")
                self.mock = True

    def read_time(self):
        """Reads current time from the DS3231 RTC. Returns datetime object."""
        if self.mock or self.bus is None:
            return datetime.now()

        try:
            d = self.bus.read_i2c_block_data(self.addr, 0x00, 7)
            sec = bcd2dec(d[0] & 0x7F)
            minute = bcd2dec(d[1])
            if d[2] & 0x40:  # 12-hour mode
                hour = bcd2dec(d[2] & 0x1F)
                if d[2] & 0x20:  # PM
                    hour = hour % 12 + 12
                else:
                    hour = hour % 12
            else:
                hour = bcd2dec(d[2] & 0x3F)
            day = bcd2dec(d[4])
            month = bcd2dec(d[5] & 0x1F)
            year = 2000 + bcd2dec(d[6])
            return datetime(year, month, day, hour, minute, sec)
        except Exception as e:
            print(f"[RTC READ ERROR]: {e}")
            return datetime.now()

    def close(self):
        if self.bus is not None:
            try:
                self.bus.close()
            except Exception:
                pass


class DHT22Sensor:
    """
    DHT22 Temperature & Humidity Sensor on GPIO 4.
    Handles read retries and evaluates general weather conditions.
    """
    def __init__(self, pin=PIN_DHT22_DATA, mock=False):
        self.mock = mock or MOCK_HARDWARE or (adafruit_dht is None)
        self.dht = None

        if not self.mock:
            try:
                board_pin = getattr(board, f"D{pin}", None)
                if board_pin is not None:
                    self.dht = adafruit_dht.DHT22(board_pin)
                    print(f"[DHT22 SENSOR]: Initialized on GPIO {pin}")
                else:
                    self.mock = True
            except Exception as e:
                print(f"[DHT22 WARNING]: Failed initializing DHT22 ({e}). Using mock.")
                self.mock = True

    def read_readings(self, retries=3):
        """
        Attempts to read temperature (Celsius) and humidity (%) from DHT22.
        Retries up to `retries` times to handle transient 1-wire timing errors.
        Returns (temp_c, humidity) or (None, None) on complete failure.
        """
        if self.mock or self.dht is None:
            return 24.5, 58.0

        for attempt in range(1, retries + 1):
            try:
                t = self.dht.temperature
                h = self.dht.humidity
                if t is not None and h is not None:
                    return float(t), float(h)
            except RuntimeError as e:
                time.sleep(0.5)
            except Exception as e:
                print(f"[DHT22 UNEXPECTED ERROR]: {e}")
                time.sleep(0.5)

        return None, None

    @staticmethod
    def evaluate_weather_condition(temp_c, humidity):
        """
        Simple, documented classification rule based on temperature & humidity:
        - If Temp >= 30°C and Humidity >= 70%: 'hot_humid' / 'hot and humid'
        - If Temp >= 30°C and Humidity < 70%:  'hot_dry'   / 'hot and dry'
        - If Temp <= 20°C and Humidity >= 70%: 'cool_humid'/ 'cool and humid'
        - If Temp <= 20°C and Humidity < 70%:  'cool_dry'  / 'cool and dry'
        - Otherwise (21°C - 29°C):             'comfortable' / 'comfortable'
        """
        if temp_c is None or humidity is None:
            return "unknown", "unknown"

        if temp_c >= 30.0:
            if humidity >= 70.0:
                return "hot_humid", "hot and humid"
            else:
                return "hot_dry", "hot and dry"
        elif temp_c <= 20.0:
            if humidity >= 70.0:
                return "cool_humid", "cool and humid"
            else:
                return "cool_dry", "cool and dry"
        else:
            return "comfortable", "comfortable"

    def close(self):
        if not self.mock and self.dht is not None:
            try:
                self.dht.exit()
            except Exception:
                pass
