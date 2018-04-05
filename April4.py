import numpy as np

a = ([['Rhia',10,20,30,40,50],
           ['Alan',75,80,75,85,100],
           ['Smith',80,80,80,90,95]])
matrix_rahul=np.matrix(a)
add_value=[[74],[90],[83]]
matrix_rahul=np.append(a,add_value,axis=1)
print(matrix_rahul)


array_value=np.array(a)
for i in a:
    print(i[:2])
    
#update the row
a[2]= ['Sam',82,79,88,97,99]
print(a)
list_value=a[0]
list_value[4]=95
print(a)    
append_value = np.array([[73], [80],[85]])
print(np.append(a, append_value, axis=1))



import re
string_value="#Python is an interpreted high level programming language for general-purpose programming*."
print(string_value)
string_list=string_value.split(" ")
print(string_list)
global_list=[]
count=0
special_char=0
for i in string_list:
    if i==i[::-1]:
        count=count+1
        

for i in string_value:
    if re.match("^[a-zA-Z0-9_]", i):
        continue
    else:
       special_char=special_char+1 

my_dict = {i:string_list.count(i) for i in string_list}
print(my_dict)

A = {5, 3, 8, 6, 1}
B = {1, 5, 3, 4, 2}
print("Set A",A)
print("Set B",B)
print(A.intersection(B))
print(A.union(B))
print(A.difference(B))
print(B.difference(A))
print(max(A))
print(min(A))
print(max(B))
print(min(B))


for i in range(900,1001):
    for j in range(2,i):
        if i%j==0:
            break
    if int(i)==int(j+1):
        print("The prime number",i)
        if str(i)==str(i)[::-1]:
            print("Prime Palindrome number",i)

  
