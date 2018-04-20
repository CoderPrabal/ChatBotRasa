#adult.data
"""import pandas as pd
import numpy as np
from sklearn import preprocessing
file_obj=pd.read_csv("adult.data.csv")
file_obj.columns=range(0,len(file_obj.columns))
le=preprocessing.Lab
for i in range(0,len(file_obj.dtypes)):
    if file_obj[i]==object:
        le.fit(file_obj)"""
"""print(file_obj)
print(file_obj.dtypes)
print(file_obj.columns)
file_obj.replace(" ?",np.NaN,inplace=True)
print(file_obj["workclass"])
file_obj["age"].fillna(file_obj["age"].mean())
file_obj["workclass"].fillna(file_obj["workclass"].value_counts())
print("The count is")
print(file_obj["age"].value_counts(dropna=False))"""

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn import preprocessing
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
df = pd.read_csv("adult.data.csv")
df.columns = range(0,len(df.columns))
df = df.replace(' ?',np.NaN)
df = df.replace(' <=50K',0)
df = df.replace(' >50K',1)
df = df.dropna()
y=df[12]
df = df.drop(12,axis=1)
le = preprocessing.LabelEncoder()
df=df.apply(preprocessing.LabelEncoder().fit_transform)
x_train,x_test,y_train,y_test=train_test_split(df,y,test_size=0.6,random_state=42)
kmeans = KMeans(n_clusters=2)  
kmeans.fit(x_train)
result=kmeans.predict(x_test)  
kmeans.cluster_centers_
print(accuracy_score(y_test,result))
x = kmeans.labels_  
