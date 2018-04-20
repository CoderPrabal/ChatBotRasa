import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv('C:\\Users\\user\\.spyder-py3\\housing.csv')
print(df)
RM=df['RM']
LSTAT=df['LSTAT']
PTRATIO=df['PTRATIO']
MEDV=df['MEDV']
print(df.describe())
print(df.corr())
p1=np.polyfit(RM,MEDV,1)
plt.scatter(RM,MEDV)
plt.plot(RM,np.polyval(p1,RM),'r-')
plt.xlabel('RM')
plt.ylabel('MEDV')
plt.show()
p1=np.polyfit(LSTAT,MEDV,1)
plt.scatter(LSTAT,MEDV)
plt.plot(LSTAT,np.polyval(p1,LSTAT),'r-')
plt.xlabel('LSTAT')
plt.ylabel('MEDV')
plt.show()
p1=np.polyfit(PTRATIO,MEDV,1)
plt.scatter(PTRATIO,MEDV)
plt.plot(PTRATIO,np.polyval(p1,PTRATIO),'r-')
plt.xlabel('PTRATIO')
plt.ylabel('MEDV')
plt.show()