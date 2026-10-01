from gpiozero import Buzzer
from time import sleep

buzzer = Buzzer(5)
    
for _ in range(2):
    for _ in range(5):
        buzzer.on()
        sleep(0.5)
        buzzer.off()
        sleep(0.5)
    sleep(0.5)
