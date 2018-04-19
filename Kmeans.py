import math
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
cluster_1=[]
cluster_2=[]
centroid_for_cluster1=feature_dict[1]
centroid_for_cluster2=feature_dict[4]

def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))
temp_count=True

while temp_count :
    print("Inside While============================================================")
    sum_value_x=0
    sum_value_y=0
    prev_cluster_1=cluster_1
    prev_cluster_2=cluster_2
    
    for ele,value in feature_dict.items():
        print("Inside first for ==============================================")
        element=feature_dict[ele]
        print("The key is",ele)
        print("The element is",element)
        check_dist_cent_1=euclidean_dist(element[0],element[1],centroid_for_cluster1[0],centroid_for_cluster1[1])
        print(check_dist_cent_1)
        check_dist_cent_2=euclidean_dist(element[0],element[1],centroid_for_cluster2[0],centroid_for_cluster2[1])
        print(check_dist_cent_2)
        if (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) :
            cluster_1.append(ele)
        elif (check_dist_cent_2<check_dist_cent_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        elif (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        else :
            print("Please check you r program once")
    print(cluster_1)
    print(cluster_2)
    #count=count+1
   
    if cluster_1==prev_cluster_1 and cluster_2==prev_cluster_2:
         temp_count=False
    
    for i in cluster_1:
        print("Inside second for ==============================================")
        print("key",i)
        element=feature_dict[i]
        print("Element",element)
        sum_value_x=sum_value_x+element[0]
        print(sum_value_x)
        sum_value_y=sum_value_y+element[1]
        print(sum_value_y)
    print("Final Value",sum_value_x/(len(cluster_1)))
    print("Final Value",sum_value_y/(len(cluster_1)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_1)),sum_value_y/(len(cluster_1))])  
    
    sum_value_x=0
    sum_value_y=0
    
    for i in cluster_2:
        print("Inside third for ==============================================")
        element=feature_dict[i]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
        
    print("Final Value",sum_value_x/(len(cluster_2)))
    print("Final Value",sum_value_y/(len(cluster_2)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_2)),sum_value_y/(len(cluster_2))])  
    
print("The cluster 1",cluster_1)
print("The cluster 2",cluster_2)
