from numpy import linspace,cumsum,array
import matplotlib.pyplot as plt
import scipy.signal as signal


### PARAMETERS

fs=256 #Hz
lowpass=5#Hz
highpass=100 #Hz
order=1 #Bandpass filter order
window_size = 5 # For median filtering

def Import_Data(filename):
    f=open(filename)
    data=f.read()
    temp=data.split('\n')
    temp.remove(temp[len(temp)-1])
    temp=list(map(int,temp))
    average_data = int(sum(temp) / len(temp))
    data_rect = array(temp) - average_data # remove the average for PSD correct calculation
    return data_rect

def PSD(data,fs):
    f, Pxx_den = signal.welch(data, fs, nperseg=1024)
    f1 = plt.figure()
    ax1 = f1.add_subplot(111)
    ax1.plot(f,Pxx_den)
    plt.xlabel('frequency [Hz]')
    plt.ylabel('PSD')

def Band_Pass_filter(low_cut,high_cut,order,fs,data):
    nyq = 0.5 * fs
    low = low_cut / nyq
    high=high_cut/nyq
    b,a=signal.butter(order,[low,high],btype='bandpass',analog='False')
    y_1 = signal.lfilter(b, a, data)
    return y_1





if "__name__==__main__":
    data = Import_Data('your_file.txt')
    # PSD(data,fs)
    band_pass_filtered_data=Band_Pass_filter(lowpass,highpass,order,fs,data)
    median_filtered_data=signal.medfilt(band_pass_filtered_data,window_size) # median filter preserves edges better than the moving average filter

    PSD(data,fs)
    plt.show()
    # PSD(band_pass_filtered_data,fs)
    # plt.show()



    # x = linspace(0, 1, len(data))
    # plt.plot(x,data)
    # plt.plot(x,band_pass_filtered_data)
    # plt.plot(x,median_filtered_data)
    # plt.show()