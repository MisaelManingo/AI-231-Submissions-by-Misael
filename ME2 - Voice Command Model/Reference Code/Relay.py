from gpiozero import OutputDevice
from time import sleep

relay = OutputDevice(17, active_high=False, initial_value=False)

try:
    while True:
        relay.on()
        print("RELAY ON")
        sleep(1)
        relay.off()
        print("Relay OFF")
        sleep(1)
except KeyboardInterrupt:
    pass
finally:
    relay.off()
    relay.close()