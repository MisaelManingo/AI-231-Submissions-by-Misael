#!/usr/bin/env python3
"""
Mock Hardware Stubs for Dry-Run & Simulation
Allows testing the entire Voice Assistant pipeline on hosts without physical GPIO/I2C/SPI.
"""

from datetime import datetime

class MockRGBLED:
    def __init__(self, red=17, green=27, blue=22):
        self.color = (0, 0, 0)
    def off(self):
        self.color = (0, 0, 0)
    def close(self):
        pass

class MockPi5Neo:
    def __init__(self, dev, num_leds, freq):
        self.leds = [(0, 0, 0)] * num_leds
    def set_led_color(self, idx, r, g, b):
        if 0 <= idx < len(self.leds):
            self.leds[idx] = (r, g, b)
    def fill_strip(self, r, g, b):
        self.leds = [(r, g, b)] * len(self.leds)
    def update_strip(self):
        pass

class MockLCD:
    def __init__(self, *args, **kwargs):
        self.cursor_pos = (0, 0)
        self.backlight_enabled = True
    def clear(self):
        pass
    def write_string(self, s):
        pass

class MockTM1637:
    def __init__(self, clk=23, dio=24):
        pass
    def brightness(self, val=5):
        pass
    def temperature(self, val):
        pass
    def numbers(self, h, m, colon=True):
        pass
    def write(self, segs):
        pass

class MockDHT22:
    def __init__(self, pin=4):
        self.temperature = 24.0
        self.humidity = 60.0
    def exit(self):
        pass

class MockRTC:
    def read_time(self):
        return datetime.now()
    def close(self):
        pass

class MockBuzzer:
    def __init__(self, pin=6):
        pass
    def on(self):
        pass
    def off(self):
        pass
    def close(self):
        pass

class MockButton:
    def __init__(self, pin=5):
        self.when_pressed = None
    def close(self):
        pass

