#!/usr/bin/env python3
"""
Hardware Configuration & Constants for Edge Voice Assistant (Raspberry Pi 5)
Centralized configuration for GPIO pins, I2C addresses, telephony defaults,
and file storage paths.
"""

import os

# Base directory for the ME2 project
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------
# MOCK / DRY-RUN CONFIGURATION
# ---------------------------------------------------------
# Set MOCK_HARDWARE = True (via flag or environment) to simulate hardware on host
MOCK_HARDWARE = os.getenv("MOCK_HARDWARE", "0") == "1"

# ---------------------------------------------------------
# PIN MAP TABLE (All 3.3V Logic)
# ---------------------------------------------------------
# I2C Bus 1 (Physical Pins 3 & 5: SDA=GPIO 2, SCL=GPIO 3)
I2C_BUS = 1
ADDR_LCD = 0x27   # 20x4 PCF8574 I2C LCD
ADDR_RTC = 0x68   # DS3231 Real-Time Clock

# DHT22 Temperature & Humidity Sensor (Moved to GPIO 4 to avoid RGB LED conflict)
PIN_DHT22_DATA = 4       # Physical Pin 7

# Single RGB Status LED (Common Cathode to Ground)
PIN_RGB_RED = 17         # Physical Pin 11
PIN_RGB_GREEN = 27       # Physical Pin 13
PIN_RGB_BLUE = 22        # Physical Pin 15

# TM1637 4-Digit 7-Segment Display
PIN_TM1637_CLK = 23      # Physical Pin 16
PIN_TM1637_DIO = 24      # Physical Pin 18

# Grove RGB LED Strip (10x WS2813 Mini NeoPixels via SPI0 MOSI)
SPI_LED_STRIP_DEV = "dev/spidev0.0"  # Physical Pin 19 (GPIO 10 MOSI)
NUM_LEDS_STRIP = 10

# Push Button (User Input / Stop Alarm & Timer Beep)
PIN_BUTTON = 5           # Physical Pin 29 (Internal Pull-Up; Active LOW)

# Active Buzzer (Timer & Notification Beep)
PIN_BUZZER = 6           # Physical Pin 31 (Moved from 5 to avoid Button conflict)

# ---------------------------------------------------------
# TELEPHONY & ADB CONSTANTS (Easily editable at the top)
# ---------------------------------------------------------
CALL_NUMBER = "+639088152097"
MESSAGE_NUMBER = "+639088152097"
MESSAGE_TEXT = "Hello from my Raspberry Pi!"
CALL_CARRIER = "Globe"   # Carrier name for SIM selection (e.g. "Globe" or "SMART")

# ---------------------------------------------------------
# FILE SYSTEM & AUDIO ASSET PATHS
# ---------------------------------------------------------
MUSIC_DIR = os.path.join(BASE_DIR, "Music")
TTS_DIR = os.path.join(BASE_DIR, "tts")
CHIME_WAKE_PATH = os.path.join(BASE_DIR, "chime_wake.wav")
CHIME_END_PATH = os.path.join(BASE_DIR, "chime_end.wav")
ALARM_SOUND_PATH = os.path.join(BASE_DIR, "Alarm.mp3")

STATE_FILE_PATH = os.path.join(BASE_DIR, "assistant_state.json")

# Default Audio Parameters
AUDIO_SAMPLE_RATE = 44100
DEFAULT_MUSIC_VOLUME = 50   # 0 to 100%
DUCKED_MUSIC_VOLUME = 15    # Volume % during command listening

