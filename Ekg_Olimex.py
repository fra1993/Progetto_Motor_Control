import serial
import binascii
import codecs
# import sys
# import numpy as np
# import matplotlib.pyplot as plt
# import scipy.signal

### PARAMETERS ###

N=100 # number of byte (every packet is 17 bytes)

def Initiate_serial_comunication(port_ID,baudrate_in):
    ser = serial.Serial(port_ID, baudrate_in,timeout=None)  # baudrate should be the same as the arduino serial port frequency and timeout none as specified by Olimex
    return ser

# print(s)
# x=binascii.hexlify(s) #from bytes to hex
# y=codecs.decode(x,'hex') #from hex to bytes
# z=x.decode()
# print(z)

if '__name__==__main':
    ser=Initiate_serial_comunication('COM3',57600)
    s = ser.read(N)
    i=0
    while i<(len(s)-1):
        if (hex(s[i])=='0xa5') & (hex(s[i+1])=='0x5a'):
            print('ciao')
        i=i+1
        # temp_int = int(temp_hex, 16)

