import serial
import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal


ser = serial.Serial('COM3') #, 38400, timeout=0,parity=serial.PARITY_EVEN, rtscts=1)

# 38400, n, 8, 1

#indica una trasmissione con baud rate di 9600, senza parità (n = none - no parity), con dato a 8 bit e 1 bit di Stop.
s = ser.read(100)
ss = s.decode("utf-8")
sss=ss.split("_")

text_file = open("Outputfake.txt", "w")
text_file.write("Purchase Amount: %s" % sss)
text_file.close()


# x=np.linspace(0,1,len(sss))/9600
# b,a=scipy.signal.butter(5,0.9)
# y=scipy.signal.lfilter(b,a,sss)
# plt.plot(x,y)
# plt.show()



