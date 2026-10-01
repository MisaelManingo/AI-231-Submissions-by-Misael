from RPLCD.i2c import CharLCD
from time import sleep

lcd = CharLCD('PCF8574', 0x27, cols=20, rows=4)

lcd.clear()
lcd.write_string('Hello, Pi 5!')
lcd.crlf()
lcd.write_string('Relay: OFF')
lcd.crlf()
lcd.write_string('12345678901234567890')
lcd.crlf()
lcd.write_string('         1         2')
sleep(5)

lcd.clear()
lcd.backlight_enabled = False