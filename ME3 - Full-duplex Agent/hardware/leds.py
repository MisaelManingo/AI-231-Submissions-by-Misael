#!/usr/bin/env python3
"""
LED Controllers for Raspberry Pi 5
1. Single RGB Status LED (gpiozero RGBLED on GPIO 17, 27, 22)
   - Yellow = Idle / Listening
   - Blue   = Thinking / Processing / MCP tool active
   - Green  = Speaking / Outputting voice
   - Red    = Error / Interrupted
2. Grove RGB LED Strip (10x WS2813 Mini NeoPixels via Pi5Neo on SPI0 MOSI)
   - Turn on / off
   - Brightness (0 to 100%)
   - Color (RED, GREEN, BLUE, WHITE, YELLOW, CYAN, PURPLE)
"""

from config import (
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


# Match the GUI screen colours (idle white / listening red / thinking violet / speaking teal).
# Set ME3_LED_MATCH_GUI=0 to restore the old yellow/blue/green scheme. Values are PWM duty (0..1),
# tuned by eye-guess for a common-cathode RGB LED; adjust here if the LED looks off.
import os as _os
_MATCH_GUI = _os.getenv("ME3_LED_MATCH_GUI", "1") != "0"
_LED_IDLE = (1, 1, 1) if _MATCH_GUI else (1, 1, 0)
_LED_LISTENING = (1, 0, 0.15) if _MATCH_GUI else (1, 1, 0)
_LED_THINKING = (0.45, 0, 1) if _MATCH_GUI else (0, 0, 1)
_LED_SPEAKING = (0, 0.75, 0.6) if _MATCH_GUI else (0, 1, 0)


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

    def set_listening(self):
        """User speaking (GUI: red). Previously shared set_yellow()."""
        self.current_state = "LISTENING"
        if not self.mock and self.led is not None:
            try:
                self.led.color = _LED_LISTENING
            except Exception:
                pass

    def set_yellow(self):
        """Idle / standing by (name kept for compatibility; GUI scheme: white)."""
        self.current_state = "YELLOW (Listening)"
        if not self.mock and self.led is not None:
            try:
                self.led.color = _LED_IDLE
            except Exception:
                pass

    def set_blue(self):
        """Thinking / Processing / Tool calling."""
        self.current_state = "BLUE (Thinking)"
        if not self.mock and self.led is not None:
            try:
                self.led.color = _LED_THINKING
            except Exception:
                pass

    def set_green(self):
        """Speaking / Playing response."""
        self.current_state = "GREEN (Speaking)"
        if not self.mock and self.led is not None:
            try:
                self.led.color = _LED_SPEAKING
            except Exception:
                pass

    def set_red(self):
        """Error / Barged in / Alert."""
        self.current_state = "RED (Alert/Error)"
        if not self.mock and self.led is not None:
            try:
                self.led.color = (1, 0, 0)
            except Exception:
                pass

    def off(self):
        self.current_state = "OFF"
        if not self.mock and self.led is not None:
            try:
                self.led.off()
            except Exception:
                pass

    def close(self):
        self.off()
        if not self.mock and self.led is not None:
            try:
                self.led.close()
            except Exception:
                pass


class GroveLEDStrip:
    """
    Controls 10x WS2813 NeoPixels via SPI0.
    Brightness is scaled both by value and number of lit LEDs.
    """
    COLOR_MAP = {
        "RED": (255, 0, 0),
        "GREEN": (0, 255, 0),
        "BLUE": (0, 0, 255),
        "WHITE": (255, 255, 255),
        "YELLOW": (255, 200, 0),
        "CYAN": (0, 255, 255),
        "PURPLE": (180, 0, 255),
        "ORANGE": (255, 100, 0),
    }

    def __init__(self, spi_dev=SPI_LED_STRIP_DEV, num_leds=NUM_LEDS_STRIP, mock=False):
        self.mock = mock or MOCK_HARDWARE or (Pi5Neo is None)
        self.num_leds = num_leds
        self.strip = None

        self.power = False
        self.brightness = 100
        self.color_name = "RED"

        if not self.mock:
            try:
                self.strip = Pi5Neo(spi_dev, num_leds, 800)
                self.clear()
                print(f"[GROVE LED STRIP]: Initialized {num_leds} NeoPixels on {spi_dev}")
            except Exception as e:
                print(f"[GROVE LED STRIP WARNING]: Could not open SPI LED strip ({e}). Using mock.")
                self.mock = True

    def _render(self):
        rgb = self.COLOR_MAP.get(self.color_name.upper(), (255, 255, 255))
        # Scale color by brightness
        scale = max(0.0, min(1.0, self.brightness / 100.0))
        r = int(rgb[0] * scale)
        g = int(rgb[1] * scale)
        b = int(rgb[2] * scale)

        num_lit = max(1, int(round(self.num_leds * scale))) if (scale > 0 and self.power) else 0

        print(f"💡 [LED STRIP]: Power={self.power} | Color={self.color_name} | Brightness={self.brightness}% ({num_lit}/{self.num_leds} lit)")

        if not self.mock and self.strip is not None:
            try:
                for i in range(self.num_leds):
                    if self.power and i < num_lit:
                        self.strip.set_led_color(i, r, g, b)
                    else:
                        self.strip.set_led_color(i, 0, 0, 0)
                self.strip.update_strip()
            except Exception as e:
                print(f"[LED STRIP ERROR]: Failed updating LEDs: {e}")

    def turn_on(self):
        self.power = True
        self._render()

    def turn_off(self):
        self.power = False
        self._render()

    def set_brightness(self, percent: int):
        self.brightness = max(0, min(100, int(percent)))
        if self.power:
            self._render()

    def set_color(self, color_name: str):
        color_clean = color_name.upper().strip()
        if color_clean in self.COLOR_MAP:
            self.color_name = color_clean
        else:
            self.color_name = "WHITE"
        if self.power:
            self._render()

    def clear(self):
        self.power = False
        self._render()

    def close(self):
        self.clear()

