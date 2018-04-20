import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Question1
"""df=pd.read_csv("100 Sales Records.csv")
print(df)
#print the names of column
print(df.columns)
print(df[0:10].head(10))
total_profit=df['Total Profit']
#to represent total profit
plt.plot(range(len(total_profit)),total_profit,marker='o')
plt.show()
print(df['Total Profit'])
print(df[df['Total Cost']>1000000]['Item Type'])
"""
#Question2
"""a = np.zeros(shape=(4,4))
print(a)
b = a.reshape(4,2,2)
print(b)
a=a+5
b=b+5
print(a)
print(b)
a=a-2
b=b-2
print(a)
print(b)
a=a*6
b=b*6
print(a)
print(b)
"""
#Question3
"""list1=[]
for i in range(0,4):
    list1.append(input("Enter the marks of student"))

print(list1)
print(max(list1))
print("Marks of First Student  "+list1[0])
print("Marks of Second Student "+list1[1])
print(list1.sort())
for i in range(0,2):
    list1.append(input("Enter the marks of 2 more student"))
print(list1)
"""
#Question4
"""
Exam_Result={'Name':['Prabal','MK','DK','PK','CK'],
             'Score':[90,20,11,10,4],
             'No_of_attempts':[1,2,3,4,5],
             'Qualify':['y','y','n','n','n']}
d=pd.DataFrame(Exam_Result,index=['a','b','c','d','e'])
print(d)
print(d.head(4))
print(d[d['Qualify']=='y']['Name'])
print(d[(d['Score']>=20)&(d['Score']<35)]['No_of_attempts'])
"""