import pandas as pd
import seaborn as sns
from sklearn import svm
file_obj=pd.read_csv("C:\\Users\\user\\.spyder-py3\\creditcard.csv")
print(file_obj)
print(file_obj.columns)
correlation=file_obj.corr()
print(correlation)
heat_map_matrix=sns.heatmap(file_obj)
clf = svm.SVC()
clf = svm.SVC(gamma=0.001, C=100)
test_y=file_obj['Class'][0:1000]
training_y=file_obj['Class'][1000:]
test_x=file_obj[0:1000]
training_x=file_obj[1000:]
clf.fit(training_x,training_y)
print(clf.predict(test_x,test_y))