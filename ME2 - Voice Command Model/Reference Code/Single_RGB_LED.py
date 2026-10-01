from gpiozero import RGBLED
from time import sleep

# BCM numbering
led = RGBLED(red=17, green=27, blue=22)  # PWM enabled by default

colors = [
    (1, 0, 0),  # Red
    (0, 1, 0),  # Green
    (0, 0, 1),  # Blue
    (1, 1, 0),  # Yellow
    (0, 1, 1),  # Cyan
    (1, 0, 1),  # Magenta
    (1, 1, 1),  # White
]

try:
    while True:
        for c in colors:
            led.color = c  # values are 0.0 to 1.0
            sleep(1)
except KeyboardInterrupt:
    led.off()
    led.close()