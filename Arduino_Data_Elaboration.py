from numpy import linspace
import matplotlib.pyplot as plt

def Import_Data(filename):
    f = open(filename)
    data = f.read()
    data_parsed = data.split(',')
    return data_parsed

def Import_Data_Old(filename): #txt files salvati in data 04/24/2019 si importano con questa funzione
    f=open(filename)
    data=f.read()
    return data.split('\n')


if "__name__==__main__":
    data = Import_Data_Old('your_file.txt')
    x=linspace(0,1,len(data))
    print(data)
    plt.plot(x,data)
    plt.show()