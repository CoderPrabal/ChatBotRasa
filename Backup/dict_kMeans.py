import math
import random
#coding K Means

def calc_dist(x1,y1,x2,y2):
    x1_x2=math.pow(x1-x2,2)
    y1_y2=math.pow(y1-y2,2)
    total=math.sqrt(x1_x2+y1_y2)
    return total
 
dict1={'MedicineA':[1,1],'MedicineB':[2,1],'MedicineC':[4,3],'MedicineD':[5,4]}
print(dict1)
centroid=[]
cluster1=[]
cluster2=[]
list_of_rand_cluster=random.sample(list(dict1),2)
print(list_of_rand_cluster)
#intial centroid
for i in list_of_rand_cluster:
    value=dict1[i]
    centroid.append(value)
    print(value)
    
print(centroid)

while(True):
    for i in dict1:
        print(dict1[i])
        if calc_dist(dict1[i][0],dict1[i][1],centroid[0][0],centroid[0][1])>calc_dist(dict1[i][0],dict1[i][1],centroid[1][0],centroid[1][1]):
           cluster1.append(list(dict1[i][0],dict1[i][1])) 
    

