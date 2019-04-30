

from serial import Serial
import numpy as np
from math import ceil

N=1000
Packet_Dimension=19

ser = Serial('COM3', 57600 ,timeout=None)  # baudrate should be the same as the arduino serial port frequency and timeout none as specified by Olimex
s = ser.read(N)  # reading N bytes from the serial port

i=0
sw = np.zeros(ceil(N / Packet_Dimension))
sw_T=np.zeros(ceil(N / Packet_Dimension))
sw_T_index=0
sw_index = 0
while i < (len(s) - 1):
    if (hex(s[i]) == '0xa5') & (hex(s[i + 1]) == '0x5a'):
        try:
            temp = (int(hex(s[i+4]), 16)) * 256 + (int(hex(s[i+5]), 16))
            temp_T = (int(hex(s[i + 16]), 16)) * 256 + (int(hex(s[i + 17]), 16))
            initial_time = (int(hex(s[i + 18]), 16)) * 256 + (int(hex(s[i + 19]), 16))
            sw[sw_index] = temp
            sw_T[sw_T_index] = temp_T - initial_time
            sw_index = sw_index + 1
            sw_T_index = sw_T_index + 1
        except Exception as Exc:
            print(Exc)
            print('With byte string of lenght: ', len(s))
            print('Out of range index: ', i + 17)
    i = i + 1

with open('provaTimeValues.txt', 'w') as f:
     for item in sw:
         f.write("%s\n" % item)
f.close()

with open('provaTimeTime.txt', 'w') as f:
    for item in sw_T:
        f.write("%s\n" % item)
f.close()

