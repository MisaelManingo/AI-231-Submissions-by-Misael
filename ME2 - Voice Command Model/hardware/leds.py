#!/usr/bin/env python3
"""
LED Controllers for Raspberry Pi 5
1. Single RGB Status LED (gpiozero RGBLED on GPIO 17, 27, 22)
   - Yellow = Listening for wake word
   - Blue   = Listening for command
   - Green  = Understanding / executing command
   - Red    = No command executed
2. Grove RGB LED Strip (10x WS2813 Mini NeoPixels via pi5neo on SPI0 MOSI)
   - LIGHTS_ON / LIGHTS_OFF
   - BRIGHTNESS (20%, 60%, 100% lit LEDs at full intensity)
   - COLOR (RED default, GREEN, BLUE)
"""

from .config import (
    PIN_RGB_RED, PIN_RGB_GREEN, PIN_RGB_BLUE,
    SPI_LED_STRIP_DEV, NUM_LEDS_STRIP, MOCK_HARDWARE
)

try:
    from gpiozero import RGBLED
except (ImportError, Exception):
    RGBLED = None

try:
    from pi5neo import Pi5Neo
except (ImportError, Exception):
    Pi5Neo = None


class StatusRGBLED:
    """Controls the single RGB indicator LED."""
    def __init__(self, red=PIN_RGB_RED, green=PIN_RGB_GREEN, blue=PIN_RGB_BLUE, mock=False):
        self.mock = mock or MOCK_HARDWARE or (RGBLED is None)
        self.led = None
        self.current_state = "OFF"

        if not self.mock:
            try:
                self.led = RGBLED(red=red, green=green, blue=blue)
                print(f"[STATUS LED]: Initialized on Red={red}, Green={green}, Blue={blue}")
            except Exception as e:
                print(f"[STATUS LED WARNING]: Failed initializing RGB LED ({e}). Using mock.")
                self.mock = True

    def set_yellow(self):
        """Listening for wake word."""
        self.current_state = "YELLOW (Listening for Wake Word)"
        if not self.mock and self.led is not None:
            self.led.color = (1, 1, 0)

    def set_blue(self):
        """Listening for command."""
        self.current_state = "BLUE (Listening for Command)"
        if not self.mock and self.led is not None:
            self.led.color = (0, 0, 1)

    def set_green(self):
        """Understanding / executing command."""
        self.current_state = "GREEN (Executing Command)"
        if not self.mock and self.led is not None:
            self.led.color = (0, 1, 0)

    def set_red(self):
        """No command executed."""
        self.current_state = "RED (Command Ignored / Failed)"
        if not self.mock and self.led is not None:
            self.led.color = (1, 0, 0)

    def off(self):
        self.current_state = "OFF"
        if not self.mock and self.led is not None:
            self.led.off()

    def close(self):
        self.off()
        if not self.mock and self.led is not None:
            try:
                self.led.close()
            except Exception:
                pass


class GroveLEDStrip:
    """
    Controls the 10x WS2813 NeoPixel strip via SPI0.
    Brightness is mapped to the number of lit LEDs (e.g. 20% = 2 LEDs, 100% = 10 LEDs).
    Default color is RED.
    """
    COLOR_MAP = {
        "RED": (255, 0, 0),
        "GREEN": (0, 255, 0),
        "BLUE": (0, 0, 255),
        "WHITE": (255, 255, 255)
    }

    def __init__(self, spi_dev=SPI_LED_STRIP_DEV, num_leds=NUM_LEDS_STRIP, mock=False):
        self.mock = mock or MOCK_HARDWARE or (Pi5Neo is None)
        self.num_leds = num_leds
        self.strip = None

        self.power = False
        self.brightness = 100   # 20, 60, or 100%
        self.color_name = "RED" # Default is RED

        if not self.mock:
            try:
                self.strip = Pi5Neo(spi_dev, num_leds, 800)
                self.clear()
                print(f"[GROVE LED STRIP]: Initialized {num_leds} NeoPixels on {spi_dev}")
            except Exception as e:
                print(f"[GROVE LED STRIP WARNING]: Could not open SPI LED strip ({e}). Using mock.")
                self.mock = True

    def _render(self):
        rgb = self.COLOR_MAP.get(self.color_name, (255, 0, 0))
        num_lit = max(0, min(self.num_leds, int(round(self.num_leds * (self.brightness / 100.0)))))

        if not self.power:
            num_lit = 0

        print(f"💡 [GROVE STRIP]: Power={self.power} | Color={self.color_name} | Lit={num_lit}/{self.num_leds}")

        if not self.mock and self.strip is not None:
            try:
                for i in range(self.num_leds):
                    if i < num_lit:
                        self.strip.set_led_color(i, rgb[0], rgb[1], rgb[2])
                    else:
                        self.strip.set_led_color(i, 0, 0, 0)
                self.strip.update_strip()
            except Exception as e:
                print(f"[GROVE STRIP ERROR]: Failed updating LEDs: {e}")

    def turn_on(self):
        self.power = True
        self._render()

    def turn_off(self):
        self.power = False
        self._render()

    def set_brightness(self, percent):
        """Sets brightness level (20, 60, 100)."""
        self.brightness = max(0, min(100, int(percent)))
        if self.power:
            self._render()

    def set_color(self, color_name):
        """Sets color (RED, GREEN, BLUE)."""
        color_key = color_name.upper()
        if color_key in self.COLOR_MAP:
            self.color_name = color_key
            if self.power:
                self._render()

    def clear(self):
        if not self.mock and self.strip is not None:
            try:
                self.strip.fill_strip(0, 0, 0)
                self.strip.update_strip()
            except Exception:
                pass

    def close(self):
        self.clear()
