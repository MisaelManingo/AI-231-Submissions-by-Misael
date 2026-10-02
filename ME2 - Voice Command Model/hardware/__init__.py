"""
Hardware package for Raspberry Pi 5 Edge Voice Assistant
"""

from .config import (
    CALL_NUMBER, MESSAGE_NUMBER, MESSAGE_TEXT, CALL_CARRIER,
    PIN_DHT22_DATA, PIN_RGB_RED, PIN_RGB_GREEN, PIN_RGB_BLUE,
    PIN_TM1637_CLK, PIN_TM1637_DIO, PIN_BUTTON, PIN_BUZZER,
    I2C_BUS, ADDR_LCD, ADDR_RTC, SPI_LED_STRIP_DEV, NUM_LEDS_STRIP
)
from .manager import HardwareManager

__all__ = [
    "HardwareManager",
    "CALL_NUMBER",
    "MESSAGE_NUMBER",
    "MESSAGE_TEXT",
    "CALL_CARRIER"
]

