import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_excel('C:\\Users\\user\\.spyder-py3\\beer.xlsx', sheet_name='Sheet1')
df_training=df[26:52]
print("Column headings:")
print(df_training)
#print(df.columns)

p12 = df_training['PRICE 12PK']
p18 = df_training['PRICE 18PK']
p30 = df_training['PRICE 30PK']
c12 = df_training['CASES 12PK']
c18 = df_training['CASES 18PK']
c30 = df_training['CASES 30PK']

p12_training,p18_training,p30_training=p12[26:52],p18[26:52],p30[26:52]
c12_training,c18_training,c30_training=c12[26:52],c18[26:52],c30[26:52]
print(df_training.corr())
test=df_training.corr()
p12_c12=test['PRICE 12PK']['CASES 12PK']
p18_c18=test['PRICE 18PK']['CASES 18PK']
p30_c30=test['PRICE 30PK']['CASES 18PK']
print(p12_c12)
print(p18_c18)
print(p30_c30)
p1=np.polyfit(p12,c12,1)
p2=np.polyfit(p18,c18,1)
p3=np.polyfit(p30,c30,1)
print(p1)
print(p2)
print(p3)
df_predict=df[0:26]
print(df_predict)
pred_c12=[]
pred_c18=[]
pred_c30=[]
p12_pred = np.array(df_predict['PRICE 12PK'])
p18_pred= np.array(df_predict['PRICE 18PK'])
p30_pred= np.array(df_predict['PRICE 30PK'])
c12 = df_predict['CASES 12PK']
c18 = df_predict['CASES 18PK']
c30 = df_predict['CASES 30PK']

for i in p12_pred:
    pred_c12.append((i*p1[0])+p1[1])
    
for i in p18_pred:
    pred_c18.append((i*p2[0])+p2[1])

for i in p30_pred:
    pred_c30.append((i*p3[0])+p3[1])


print(pred_c12)
plt.plot(list(range(26)),c12)
plt.plot(list(range(26)),pred_c12,marker='^')
plt.plot(df_predict['PRICE 12PK'],pred_c12,marker='o')
plt.xlabel("week")
plt.ylabel("sales")
plt.legend(('actual','predict'))
plt.show()

print(pred_c18)
plt.plot(list(range(26)),c18)
plt.plot(list(range(26)),pred_c18,marker='^')
plt.plot(df_predict['PRICE 12PK'],pred_c18,marker='o')
plt.xlabel("week")
plt.ylabel("sales")
plt.legend(('actual','predict'))
plt.show()
#print(p12_training)
#print(p18_training)
#print(p30_training)
print(pred_c30)
plt.plot(list(range(26)),c30)
plt.plot(list(range(26)),pred_c30,marker='^')
plt.plot(df_predict['PRICE 12PK'],pred_c30,marker='o')
plt.xlabel("week")
plt.ylabel("sales")
plt.legend(('actual','predict'))
plt.show()


#########


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_excel('C:\\Users\\user\\.spyder-py3\\beer.xlsx', sheet_name='Sheet1')
df_training=df[0:26]
print("Column headings:")
print(df_training)
#print(df.columns)

p12 = df_training['PRICE 12PK']
p18 = df_training['PRICE 18PK']
p30 = df_training['PRICE 30PK']
c12 = df_training['CASES 12PK']
c18 = df_training['CASES 18PK']
c30 = df_training['CASES 30PK']

p12_training,p18_training,p30_training=p12[26:52],p18[26:52],p30[26:52]
c12_training,c18_training,c30_training=c12[26:52],c18[26:52],c30[26:52]
print(df_training.corr())
test=df_training.corr()
p12_c12=test['PRICE 12PK']['CASES 12PK']
p18_c18=test['PRICE 18PK']['CASES 18PK']
p30_c30=test['PRICE 30PK']['CASES 18PK']
print(p12_c12)
print(p18_c18)
print(p30_c30)
p1=np.polyfit(p12,c12,1)
p2=np.polyfit(p18,c18,1)
p3=np.polyfit(p30,c30,1)
print(p1)
print(p2)
print(p3)
df_predict=df[26:52]
print(df_predict)
pred_c12=[]
pred_c18=[]
pred_c30=[]
p12_pred = np.array(df_predict['PRICE 12PK'])
p18_pred= np.array(df_predict['PRICE 18PK'])
p30_pred= np.array(df_predict['PRICE 30PK'])
c12 = df_predict['CASES 12PK']
c18 = df_predict['CASES 18PK']
c30 = df_predict['CASES 30PK']

for i in p12_pred:
    pred_c12.append((i*p1[0])+p1[1])
    
for i in p18_pred:
    pred_c18.append((i*p2[0])+p2[1])

for i in p30_pred:
    pred_c30.append((i*p3[0])+p3[1])


print(pred_c12)
plt.plot(list(range(26)),c12)
plt.plot(list(range(26)),pred_c12,marker='^')
plt.plot(df_predict['PRICE 12PK'],pred_c12,marker='o')
plt.xlabel("week")
plt.ylabel("sales")
plt.legend(('actual','predict'))
plt.show()

print(pred_c18)
plt.plot(list(range(26)),c18)
plt.plot(list(range(26)),pred_c18,marker='^')
plt.plot(df_predict['PRICE 12PK'],pred_c18,marker='o')
plt.xlabel("week")
plt.ylabel("sales")
plt.legend(('actual','predict'))
plt.show()
#print(p12_training)
#print(p18_training)
#print(p30_training)
print(pred_c30)
plt.plot(list(range(26)),c30)
plt.plot(list(range(26)),pred_c30,marker='^')
plt.plot(df_predict['PRICE 12PK'],pred_c30,marker='o')
plt.xlabel("week")
plt.ylabel("sales")
plt.legend(('actual','predict'))
plt.show()
mse30 = np.mean((c30 - pred_c30)**2)
mse18 = np.mean((c18 - pred_c18)**2)
mse12=np.mean((c12 - pred_c12)**2)
print(mse30)
print(mse18)
print(mse12)
