import time
import board
import adafruit_dht

dht = adafruit_dht.DHT22(board.D27)

try:
    while True:
        try:
            temperature_c = dht.temperature
            humidity = dht.humidity
            if temperature_c is not None and humidity is not None:
                temperature_f = temperature_c * 9 / 5 + 32
                print(f"Temp: {temperature_c:.1f} C / {temperature_f:.1f} F | Humidity: {humidity:.0f}%")
        except RuntimeError as e:
            print(f"Read error: {e.args[0]}")
            time.sleep(2.0)
            continue
        
        time.sleep(2.0)

except KeyboardInterrupt:
    pass
finally:
    dht.exit()