import math
import random
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
k=int(input("Enter the value for clusters"))
centroid_list=[]
general_cluster=[]
list_of_rand_cluster=random.sample(list(feature_dict),k)
for i in list_of_rand_cluster:
    general_cluster.append([i])
    element=feature_dict[i]
    centroid_list.append(element)
print("Random cluster")    
print(list_of_rand_cluster)
print("List of classes")
print(general_cluster)
print("Centroid Value")
print(centroid_list)
def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))
temp_count=True

while temp_count :
    print("Inside While============================================================")
    sum_value_x=0
    sum_value_y=0
    prev_general_cluster=general_cluster
    for ele,value in feature_dict.items():
        #print("Inside first for ==============================================")
        print("The key is",ele)
        print("The element is",value)
        distance=0
        index=0
        count_dist=0
        for i in range(0,len(centroid_list)):
            print("The centroid value")
            centroid_value=centroid_list[i]
            print(centroid_value)
            if count_dist==0:
                print("Inside if condtion")
                index=i
                print("Index")
                print(index)
                distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
                print("The distance is",distance)
                count_dist=count_dist+1
                print("Count Value")
                print(count_dist)
            else:
                print("Inside else")
                if distance>euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1]):
                    print("Inside else if")
                    distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
                    print("The distance is")
                    print(distance)
                    print("index")
                    index=i
                    print(index)
        if ele not in general_cluster[index]:       
            general_cluster[index].append(ele)
        
    if prev_general_cluster==general_cluster:
       temp_count=False
       
    temp_index=0
    centroid_list_new=[]
    for i in general_cluster:
        print("The value of i")
        print(i)
        for j in i:
            print("The value of j")
            print(j)
            print("The value of")
            value_of=feature_dict[j]
            print(value_of)
            print("sum_value_x")
            sum_value_x=sum_value_x+value_of[0]
            print(sum_value_x)
            sum_value_y=sum_value_y+value_of[1]
            print("sum_value_y")
            print(sum_value_y)
            print("The value of list",[sum_value_x/(len(i)),sum_value_y/(len(i))])
        centroid_list_new.append([sum_value_x/(len(i)),sum_value_y/(len(i))])    
            
    centroid_list=centroid_list_new
      
print(general_cluster)