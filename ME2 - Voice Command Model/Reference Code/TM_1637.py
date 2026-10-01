import time
import tm1637

CLK = 23
DIO = 24

tm = tm1637.TM1637(clk=CLK, dio=DIO)
tm.brightness(val=5)	# 0-7

tm.show("HELO")
time.sleep(2)

tm.number(1234)
time.sleep(2)

tm.numbers(12, 34)		# shows "12:34" with the colon
time.sleep(2)

tm.temperature(25)		# shows "25*C"
time.sleep(2)

tm.scroll("Raspberry Pi", delay=250)

# blinking colon
colon = True
for _ in range(10):
    t = time.localtime()
    tm.numbers(t.tm_hour, t.tm_min, colon=colon)
    colon = not colon
    time.sleep(0.5)

tm.write([0, 0, 0, 0])	# clear display