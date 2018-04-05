
import matplotlib.pyplot as plt
import statistics
import numpy as np
from scipy import stats
from statistics import variance
from scipy.stats import variation 
rent_list=[425,430,430,435,435,435,435,440,440,440,440,440,445,445,445,445,450,450,450,450,450,450,450,460,460,460,465,465,465,470,470,472,475,475,475,480,480,480,480,485,490,490,490,500,500,500,500,510,510,515,525,525,525,535,549,550,570,570,575,575,580,590,600,600,600,600,615,615]
plt.plot(rent_list)
plt.show()
mean_value=np.mean(rent_list)
print("The median is",statistics.median(rent_list))
print("The mean is",np.mean(rent_list))
print("The mode is",stats.mode(rent_list)) 
print("Cumulative frequency",stats.cumfreq(rent_list))
print("Variance from numpy is ",np.var(rent_list))
print("The variance is",variance(rent_list))
print(np.percentile(rent_list,25))
print(np.percentile(rent_list,50))
print(np.percentile(rent_list,75))
print("Frequency distribution is")
print(stats.relfreq(rent_list,numbins=10))
start_freq_value=stats.relfreq(rent_list,numbins=10)
start=start_freq_value.lowerlimit
gap=start_freq_value.binsize
print("The coffiencient of variation is")
print(variation(rent_list, axis=0)*100)
count=0
for i in start_freq_value.frequency:
    print(str(start)+"-"+str(start+gap)+"   "+str(i*70)+"   "+str(start_freq_value.frequency[count+1]))
    start+=gap
    


