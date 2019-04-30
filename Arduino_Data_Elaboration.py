from numpy import linspace,cumsum,array
import matplotlib.pyplot as plt
import scipy.signal as signal


### PARAMETERS
figures_param=1
fs=500 #Hz
lowpass=5 #Hz
highpass=100 #Hz
order=1 #Bandpass filter order
window_size = 5 # For median filtering

def Plot_figure():
    global figures_param
    plt.figure(figures_param)
    figures_param+=1

def Import_Data(filename):
    f=open(filename)
    data=f.read()
    temp=data.split('\n')
    temp.remove(temp[len(temp)-1])
    temp=list(map(int,temp))
    data_arr= array(temp)# remove the average for PSD correct calculation
    return data_arr

def PSD(data,fs):
    average_data = int(sum(data) / len(data))
    data_rect=data-average_data
    Plot_figure()
    f, Pxx_den = signal.welch(data_rect, fs, nperseg=1024)
    plt.plot(f,Pxx_den)
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

    # band_pass_filtered_data=Band_Pass_filter(lowpass,highpass,order,fs,data)

    median_filtered_data=signal.medfilt(data,window_size) # median filter preserves edges better than the moving average filter

    PSD(data,fs)
    # PSD(band_pass_filtered_data,fs)




    x = linspace(0, 1, len(data))
    Plot_figure()
    plt.plot(x,data)
    # #plt.plot(x,band_pass_filtered_data)
    # plt.plot(x,median_filtered_data)
    plt.show()