#Atul_spam_data read csv
import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
file_obj=pd.read_csv("Atul_spam_data.csv",header=0)
print(file_obj)
print(file_obj.columns)
num_features = 48
data=np.array(file_obj)
X = [data[i][:num_features] for i in range(len(data))]
y = [int(data[i][-1]) for i in range(len(data))]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)