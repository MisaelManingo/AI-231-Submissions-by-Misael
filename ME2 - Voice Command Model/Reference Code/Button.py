from gpiozero import Button
from signal import pause

button = Button(5)  # internal pull-up enabled

button.when_pressed = lambda: print("Pressed")
button.when_released = lambda: print("Released")

pause()