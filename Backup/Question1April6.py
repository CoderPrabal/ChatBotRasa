import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import variation 

list_of_data=[425,430,430,435,435,	435,435,435,440,440
,440,440,440,445,	445,445,445,445,450,450
,450,450,450,450,450,460,460,460,465,465
,465,470,470,472,475,475,475,480,480,480
,480,485,490,490,490,500,500,500,500,510
,510,515,525,525,525,535,549,550,570,570
,575,575,580,590,600,600,600,600,615,615]
print(list_of_data)
print(len(list_of_data))
plt.scatter(list_of_data,list(range((len(list_of_data)))))
plt.show()
plt.hist(list_of_data,bins=2)
plt.show()
start_freq_value=stats.relfreq(list_of_data,numbins=10)
start=start_freq_value.lowerlimit
gap=start_freq_value.binsize
print("The coffiencient of variation is")
print(variation(list_of_data, axis=0)*100)
count=0
for i in start_freq_value.frequency:
    print(str(start)+"-"+str(start+gap)+"   "+str(i*20)+"   "+str(start_freq_value.frequency[count+1]))
    start+=gap
plt.show()
