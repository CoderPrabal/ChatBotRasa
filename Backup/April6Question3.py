import matplotlib.pyplot as plt
import statistics
import numpy as np
from scipy import stats
from statistics import variance
from scipy.stats import variation 
list_of_data=[1.45,2.20,0.75,1.23,1.25,1.25,3.09,1.99,2.00,0.78,1.32,2.25,3.15,3.85,0.52,0.99,1.38,2.25,3.15,3.85,0.52,0.99,1.38,1.75,1.22,1.75]
print(len(list_of_data))
plt.scatter(list_of_data)
plt.hist(list_of_data,bins=10)
start_freq_value=stats.relfreq(list_of_data,numbins=16)
start=start_freq_value.lowerlimit
gap=start_freq_value.binsize
print("The coffiencient of variation is")
print(variation(list_of_data, axis=0)*100)
count=0
for i in start_freq_value.frequency:
    print(str(start)+"-"+str(start+gap)+"   "+str(i*20)+"   "+str(start_freq_value.frequency[count+1]))
    start+=gap
plt.show()