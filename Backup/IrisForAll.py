# =============================================================================
# from sklearn import datasets
# from sklearn.linear_model import LinearRegression
# from sklearn.cross_validation import train_test_split
# from sklearn.metrics import accuracy_score
# import pandas as pd
# import numpy as np
# =============================================================================
# =============================================================================
# 
# iris_dataset=datasets.load_iris()
# x=pd.DataFrame(iris_dataset.data)
# y=pd.DataFrame(iris_dataset.target)
# x.columns=iris_dataset.feature_names
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
# lr=LinearRegression()
# lr.fit(x_train,y_train)
# predict_val=lr.predict(x_test)
# score_value=lr.score(x_test,y_test)
# print(score_value)
# #sepal_length sepal_width petal_length petal_width
# print(lr.coef_)
# print(lr.intercept_)
# for i in range(0,len(y_test)):
#     print("%d vs %d"%(np.array(y_test)[i],predict_val[i]))
# 
# =============================================================================

#====================logistic Regression
# =============================================================================
# 
# from sklearn import datasets
# from sklearn.linear_model import LogisticRegression 
# from sklearn.cross_validation import train_test_split
# from sklearn.metrics import accuracy_score
# import pandas as pd
# import numpy as np
# 
# iris_dataset=datasets.load_iris()
# x=pd.DataFrame(iris_dataset.data)
# y=pd.DataFrame(iris_dataset.target)
# x.columns=iris_dataset.feature_names
# x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
# lr=LogisticRegression()
# lr.fit(x_train,y_train)
# pr=lr.predict(x_test)
# score_value=lr.score(x_test,y_test)
# print(score_value)
# print(accuracy_score(y_test,pr))
# for i in range(0,len(y_test)):
#      print("%d vs %d"%(np.array(y_test)[i],pr[i]))
"""
import numpy as np
from sklearn import svm
clf=svm.SVC()
clf.fit(x_train,y_train)
pr=clf.predict(x_test)
print(accuracy_score(y_test,pr))
for i in range(0,len(y_test)):
      print("%d vs %d"%(np.array(y_test)[i],pr[i]))
"""

# =============================================================================
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=8)
km=KMeans(n_clusters=2)
km.fit(x_train) 
pr=km.predict(x_test)
print(accuracy_score(y_test,pr))   
# =============================================================================
# from sklearn.naive_bayes import GaussianNB
# nb=GaussianNB()
# nb.fit(x_train,y_train)
# pr=nb.predict(y_test)
# print(nb.score(x_test,y_test))
# print(accuracy_score(y_test,pr))
# for i in range(0,len(y_test)):
#       print("%f vs %f"%(np.array(y_test)[i],pr[i]))
# =============================================================================
