import time
from datetime import datetime
from smbus2 import SMBus

BUS = 1
ADDR = 0x68


def bcd2dec(b):
    return (b >> 4) * 10 + (b & 0x0F)


def dec2bcd(d):
    return ((d // 10) << 4) | (d % 10)


def read_time(bus):
    d = bus.read_i2c_block_data(ADDR, 0x00, 7)
    sec = bcd2dec(d[0] & 0x7F)
    minute = bcd2dec(d[1])
    if d[2] & 0x40:  # 12-hour mode
        hour = bcd2dec(d[2] & 0x1F)
        if d[2] & 0x20:  # PM
            hour = hour % 12 + 12
        else:
            hour = hour % 12
    else:
        hour = bcd2dec(d[2] & 0x3F)
    day = bcd2dec(d[4])
    month = bcd2dec(d[5] & 0x1F)
    year = 2000 + bcd2dec(d[6])
    return datetime(year, month, day, hour, minute, sec)


def set_time(bus, dt):
    bus.write_i2c_block_data(ADDR, 0x00, [
        dec2bcd(dt.second),
        dec2bcd(dt.minute),
        dec2bcd(dt.hour),          # 24-hour mode
        dt.isoweekday(),           # 1-7
        dec2bcd(dt.day),
        dec2bcd(dt.month),
        dec2bcd(dt.year - 2000),
    ])
    # clear the Oscillator Stop Flag
    status = bus.read_byte_data(ADDR, 0x0F)
    bus.write_byte_data(ADDR, 0x0F, status & 0x7F)


def read_temp(bus):
    msb, lsb = bus.read_i2c_block_data(ADDR, 0x11, 2)
    if msb & 0x80:
        msb -= 256
    return msb + (lsb >> 6) * 0.25


if __name__ == "__main__":
    with SMBus(BUS) as bus:
        # Run once to set the RTC from the Pi's system time, then comment out:
#         set_time(bus, datetime.now())

        if bus.read_byte_data(ADDR, 0x0F) & 0x80:
            print("Warning: oscillator stopped (battery dead or time never set)")

        try:
            while True:
                now = read_time(bus)
                print(now.strftime("%Y-%m-%d %H:%M:%S"), f"| {read_temp(bus):.2f} °C")
                time.sleep(1)
        except KeyboardInterrupt:
            pass