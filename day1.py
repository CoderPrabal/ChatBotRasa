for i in range(0,3000):
    if i%9==0:
        if i%5!=0:
            print(i)



radius=int(input("Enter the value of n"))
dict={}
for i in range(1,radius):
    area=2*3.14*radius
    dict[i]=area

print(dict)


student=int(input("Enter the number of input"))
tuple1=()
for i in range(1,student):
    height = float(input("Enter the height"))
    weight = float(input("Enter the weight"))
    bmi=weight/(height*height)
    tuple1=tuple1+(bmi,)

print(tuple1)

######

string1=input("Enter the comma separated string of word")
list1=string1.split(',')
print(list1)
list1.sort()
print(list1)

#######
import numpy as np
a = np.array([[ 5, 1 ,3], [ 1, 1 ,1], [ 1, 2 ,1]])
b = np.array([1, 2, 3])
value=a*b
print(value)



#####
"""password=input("Enter the input")

print(check_digit)
list1=list(password)
print(list1)
length_value=len(list1)
if length_value>=8 and length_value<=15:
    temp_value=0
    for i in list1:
        if str(i).isdigit():
            temp_value=1
        if temp_value==0:
            break"""
#####


