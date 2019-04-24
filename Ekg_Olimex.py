from serial import Serial
from math import ceil
import numpy as np
import matplotlib.pyplot as plt

### PARAMETERS ###

N=100 # number of byte (every packet is 17 bytes)
Packet_Dimension=17



### FUNCTIONS ###


def Initiate_serial_comunication(port_ID,baudrate_in):
    ser = Serial(port_ID, baudrate_in,timeout=None)  # baudrate should be the same as the arduino serial port frequency and timeout none as specified by Olimex
    return ser

#THIS FUNCTION FINDS THE HEADER OF THE PACKET,
#TRANSFORMS BYTES IN INTEGERS RED FROM THE ARDUINO'S ADS AND STORES THEM IN AN ARRAY (SW)
def GetaData(byte_string):
    i = 0
    sw = np.zeros(ceil(N / Packet_Dimension))
    sw_index = 0
    while i < (len(s) - 1):
        if (hex(s[i]) == '0xa5') & (hex(s[i + 1]) == '0x5a'):
            try:
                temp = (int(hex(s[i + 4]), 16)) * 256 + (int(hex(s[i + 5]), 16))
                sw[sw_index] = temp
                sw_index = sw_index + 1
            except Exception as Exc:
                print(Exc)
                print('With byte string of lenght: ',len(s))
                print('Out of range index: ', i+5)
        i = i + 1
    return sw

def Save_Data_txt(filename,data):
    with open(filename, 'w') as f:
        for item in data:
            f.write("%s\n" % item)
    f.close()

### MAIN ###

if '__name__==__main':
    ser=Initiate_serial_comunication('COM3',57600) #starting comunication
    s = ser.read(N) #reading N bytes from the serial port
    data=GetaData(s) #data of voltage (int) measured from Arduino's ADC
    Save_Data_txt('prova.txt', data)









