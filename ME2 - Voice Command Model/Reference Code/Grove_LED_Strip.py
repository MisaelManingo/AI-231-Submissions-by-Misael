import time
from pi5neo import Pi5Neo

NUM_LEDS = 10
neo = Pi5Neo('dev/spidev0.0', NUM_LEDS, 800)

def wheel(pos):
    pos %= 256
    if pos < 85:
        return (255 - pos * 3, pos * 3, 0)
    if pos < 170:
        pos -= 85
        return (0, 255 - pos * 3, pos * 3)
    pos -= 170
    return (pos * 3, 0, 255 - pos * 3)

def clear():
    neo.fill_strip(0, 0, 0)
    neo.update_strip()
    
try:
    # show primary colors
    for color in [(255, 0, 0), (0, 255, 0), (0, 0, 255)]:
        neo.fill_strip(*color)
        neo.update_strip()
        time.sleep(0.7)
    
    # moving dot    
    for i in range(NUM_LEDS):
        neo.fill_strip(0, 0, 0)
        neo.set_led_color(i, 255, 100, 0)
        neo.update_strip()
        time.sleep(0.1)
        
    # rainbow
    for step in range(256 * 3):
        for i in range(NUM_LEDS):
            r, g, b = wheel((i * 256 // NUM_LEDS) + step)
            neo.set_led_color(i, r // 4, g // 4, b // 4)
        neo.update_strip()
        time.sleep(0.02)

except KeyboardInterrupt:
    pass
finally:
    clear()