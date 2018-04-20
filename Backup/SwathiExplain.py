import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df=pd.read_excel('beer.xlsx', sheet_name='Sheet1')

#df = pd.read_excel('beer.xlsx', sheetname='Sheet1')
 
print("Column headings:")
print(df.columns)
p12 = df['PRICE 12PK']
p18 = df['PRICE 18PK']
p30 = df['PRICE 30PK']
c12 = df['CASES 12PK']
c18 = df['CASES 18PK']
c30 = df['CASES 30PK']
week=range(len(p12))
plt.plot(week,p12,marker='o')
plt.show()
plt.plot(week,p18,marker='o')
plt.show()
plt.plot(week,p30,marker='o')
plt.show()
p12_norm = p12/np.max(p12)
c12_norm = c12/np.max(c12)
p1=np.polyfit(p12,c12,1)
p2=np.polyfit(p18,c18,1)
p3=np.polyfit(p30,c30,1)
print(p1)
print(p2)
print(p3)
plt.scatter(week,p12_norm)
plt.plot(p12,np.polyval(p1,p12),'r-')
a=np.polyfit(week,p12,1)
b=np.polyfit(week,p18,1)
c=np.polyfit(week,p30,1)
print(a)
print(b)
print(c)
lp12=[]
lp18=[]
lp30=[]
for i in range(53,100):
    lp12.append((a[0]*i)+a[1])
    lp18.append((b[0]*i)+b[1])
    lp30.append((c[0]*i)+c[1])
print(lp12)
print(lp18)
print(lp30)
plt.show()


