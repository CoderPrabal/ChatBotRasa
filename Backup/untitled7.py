
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_excel('beer.xlsx', sheetname='Sheet1')
 
print("Column headings:")
print(df.columns)

p12 = df['PRICE 12PK']
p18 = df['PRICE 18PK']
p30 = df['PRICE 30PK']

c12 = df['CASES 12PK']
c18 = df['CASES 18PK']
c30 = df['CASES 30PK']
#data=df.corr()
#3print(sb.heatmap(data,xticklabels="Price",yticklabels="Sales Volume"))
#print(data)
p1=np.polyfit(p12,c12,1)
print(np.polyval(p1,p12))
plt.scatter(p12,list(range(1,53)))
plt.plot(p12,np.polyval(p1,p12),'r-')
plt.show()
week1=np.polyfit(list(range(1,53)),p12,1)
print(week1)
list_of_predict_price=[]
for i in range(53,100):
    list_of_predict_price.append((i*p1[0])+p1[1])
    
print(list_of_predict_price)