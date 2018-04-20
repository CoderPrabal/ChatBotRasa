# -*- coding: utf-8 -*-
# *** Spyder Python Console History Log ***

## ---(Tue Apr  3 13:03:13 2018)---
runfile('C:/Users/user/.spyder-py3/temp.py', wdir='C:/Users/user/.spyder-py3')
492-rent_list
rent_list - 492
runfile('C:/Users/user/.spyder-py3/temp.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/day4.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/iris.py', wdir='C:/Users/user/.spyder-py3')

## ---(Thu Apr  5 17:08:49 2018)---
runfile('C:/Users/user/.spyder-py3/iris.py', wdir='C:/Users/user/.spyder-py3')

## ---(Fri Apr  6 09:41:40 2018)---
runfile('C:/Users/user/.spyder-py3/LinearRegression.py', wdir='C:/Users/user/.spyder-py3')
-0.30285714*180
-54.5142852/3.14
help(polyval)
help(np.polyval)
y
mean(y)
np.mean(y)
runfile('C:/Users/user/.spyder-py3/LinearRegression.py', wdir='C:/Users/user/.spyder-py3')
help(np.polyfit)
runfile('C:/Users/user/.spyder-py3/LinearRegression.py', wdir='C:/Users/user/.spyder-py3')
help(np.polyfit)
runfile('C:/Users/user/.spyder-py3/LinearRegression.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/Test2.py', wdir='C:/Users/user/.spyder-py3')
help(dict)
runfile('C:/Users/user/.spyder-py3/Test2.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/Test3.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/Test4.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/Test3.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/April6Question4.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/April6Question3.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/temp.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/April6Question3.py', wdir='C:/Users/user/.spyder-py3')
help(plt.hist)
runfile('C:/Users/user/.spyder-py3/April6Question3.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/Question1April6.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/April6Question4.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/Question1April6.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/April6Question3.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/Question1April6.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/untitled7.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/Beer.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/Desktop/Beer.py', wdir='C:/Users/user/Desktop')

## ---(Sat Apr  7 13:00:28 2018)---
runfile('C:/Users/user/Desktop/Beer.py', wdir='C:/Users/user/Desktop')
price_corr_mat1 = numpy.corrcoef([p12,c12,c18,c30])
runfile('C:/Users/user/Desktop/Beer.py', wdir='C:/Users/user/Desktop')
runfile('C:/Users/user/.spyder-py3/untitled7.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/Question1April6.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/SwathiExplain.py', wdir='C:/Users/user/.spyder-py3')
p12
runfile('C:/Users/user/.spyder-py3/SwathiExplain.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/Desktop/Beer.py', wdir='C:/Users/user/Desktop')
a=np.polyfit(week,p12,1)
b=np.polyfit(week,p18,1)
c=np.polyfit(week,p30,1)
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


plt.show()
for i in range(53,100):
    lp12.append((a[0]*i)+a[1])
    lp18.append((b[0]*i)+b[1])
    lp30.append((c[0]*i)+c[1])

print(lp12)
print(lp18)
print(lp30)
plt.show()
runfile('C:/Users/user/.spyder-py3/TestingLinearReg.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/Desktop/Beer.py', wdir='C:/Users/user/Desktop')
runfile('C:/Users/user/.spyder-py3/prediction.py', wdir='C:/Users/user/.spyder-py3')
df_predict['PRICE 12PK']
pred_c12
plt.plot(df_predict['PRICE 12PK'],pred_c12,marker='o')
runfile('C:/Users/user/.spyder-py3/prediction.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/tryingudacity.py', wdir='C:/Users/user/.spyder-py3')
print ("checking for nltk")
try:
    import nltk
except ImportError:
    print ("you should install nltk before continuing")


print ("checking for numpy")
try:
    import numpy
except ImportError:
    print ("you should install numpy before continuing")


print ("checking for scipy")
try:
    import scipy
except:
    print ("you should install scipy before continuing")


print ("checking for sklearn")
try:
    import sklearn
except:
    print ("you should install sklearn before continuing")


import urllib.request
url = "enron_mail_20150507.tar.gz"
#urllib.request.urlretrieve(url, filename="enron_mail_20150507.tar.gz") 
#print ("download complete!")


print()
print ("unzipping Enron dataset (this may take a while)")
import tarfile
import os
os.chdir("..")
tfile = tarfile.open(r"C://Users//user//enron_mail_20150507.tar.gz", "r:gz")
tfile.extractall(".")

print ("you're ready to go!")
runfile('C:/Users/user/.spyder-py3/tryingudacity.py', wdir='C:/Users/user/.spyder-py3')
print ("checking for nltk")
try:
    import nltk
except ImportError:
    print ("you should install nltk before continuing")


print ("checking for numpy")
try:
    import numpy
except ImportError:
    print ("you should install numpy before continuing")


print ("checking for scipy")
try:
    import scipy
except:
    print ("you should install scipy before continuing")


print ("checking for sklearn")
try:
    import sklearn
except:
    print ("you should install sklearn before continuing")


import urllib.request
url = "enron_mail_20150507.tar.gz"
#urllib.request.urlretrieve(url, filename="enron_mail_20150507.tar.gz") 
#print ("download complete!")


print()
print ("unzipping Enron dataset (this may take a while)")
import tarfile
import os
os.chdir("..")
tfile = tarfile.open(r"C:\\Users\\user\\.spyder-py3\\enron_mail_20150507.tar.gz", "r:tar")
tfile.extractall(".")

print ("you're ready to go!")
runfile('C:/Users/user/.spyder-py3/Re.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/Housingdata.py', wdir='C:/Users/user/.spyder-py3')
list1=[1,2,3,4,5]
import numpy
np.array(list1)
array[i]==5
runfile('C:/Users/user/.spyder-py3/AssignmentApril10.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/AssignSalesWork.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/AssignmentApril10.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/BankingData.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/t.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/Adult.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/salary_dataset.py', wdir='C:/Users/user/.spyder-py3')
from sklearn.cross_validation import train_test_split
help(train_test_split)
runfile('C:/Users/user/.spyder-py3/salary_dataset.py', wdir='C:/Users/user/.spyder-py3')
X
Y
help(train_test_split)
runfile('C:/Users/user/.spyder-py3/salary_dataset.py', wdir='C:/Users/user/.spyder-py3')
X_train
X_test
Y_test
y_test
y_train
from sklearn.linear_model import LinearRegression
runfile('C:/Users/user/.spyder-py3/salary_dataset.py', wdir='C:/Users/user/.spyder-py3')
X_test
Y_test
Y_pred
Y_test
runfile('C:/Users/user/.spyder-py3/salary_dataset.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/Assignment2_april11.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/salary_dataset.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/Assignment2_april11.py', wdir='C:/Users/user/.spyder-py3')
5+55+555
runfile('C:/Users/user/.spyder-py3/Assignment2_april11.py', wdir='C:/Users/user/.spyder-py3')
sum1 = 0
for i in range(0,3):
    sum1 += math.pow(10,i)*5)
sum1 = 0
for i in range(0,3):
    sum1 += math.pow(10,i)*5
    
sum1
sum1 = 0
for i in range(0,3):
    sum1 += (math.pow(10,i)*5 + sum1)
    
sum1
sum1 = 0
for i in range(0,3):
    sum1 += (math.pow(10,i)*5 + sum1)
    print(sum1)
    
sum1 = []
for i in range(0,3):
    sum1.append(math.pow(10,i)*5)
    print(sum(sum1))
    
sum1 = 0
for i in range(0,3):
    sum1 += (math.pow(10,i)*5 + sum1)
    
sum1
sum1 = []
for i in range(0,3):
    sum1.append(math.pow(10,i)*5)
    
sum1
series = []
for i in range(0,3):
    sum1 = math.pow(10,i)*5
    series.append(sum(sum1))
    
runfile('C:/Users/user/.spyder-py3/Assignment2_april11.py', wdir='C:/Users/user/.spyder-py3')
1+1/2.+1/3.+1/4.+1/5.
runfile('C:/Users/user/.spyder-py3/BankingData.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/salary_dataset.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/Test_Camera.py', wdir='C:/Users/user/.spyder-py3')
camera_dataset['Price'].describe
camera_dataset['Price'].describe()
runfile('C:/Users/user/.spyder-py3/Test_Camera.py', wdir='C:/Users/user/.spyder-py3')
camera_dataset.corr()
camera_dataset.columns
runfile('C:/Users/user/.spyder-py3/Test_Camera.py', wdir='C:/Users/user/.spyder-py3')
camera_dataset.columns
runfile('C:/Users/user/.spyder-py3/Test_Camera.py', wdir='C:/Users/user/.spyder-py3')
camera_dataset.columns
runfile('C:/Users/user/.spyder-py3/Test_Camera.py', wdir='C:/Users/user/.spyder-py3')
camera_dataset.columns
runfile('C:/Users/user/.spyder-py3/Test_Camera.py', wdir='C:/Users/user/.spyder-py3')
camera_dataset.columns
runfile('C:/Users/user/.spyder-py3/Test_Camera.py', wdir='C:/Users/user/.spyder-py3')
camera_dataset.columns
runfile('C:/Users/user/.spyder-py3/Test_Camera.py', wdir='C:/Users/user/.spyder-py3')
camera_dataset.columns
runfile('C:/Users/user/.spyder-py3/Test_Camera.py', wdir='C:/Users/user/.spyder-py3')

## ---(Fri Apr 13 09:30:58 2018)---
runfile('C:/Users/user/.spyder-py3/SVM.py', wdir='C:/Users/user/.spyder-py3')
print(digits.images[0])
runfile('C:/Users/user/.spyder-py3/SVM.py', wdir='C:/Users/user/.spyder-py3')
digits=datasets.load_digits()
type(digits)
type(digits.data)
runfile('C:/Users/user/.spyder-py3/SVM.py', wdir='C:/Users/user/.spyder-py3')
dig_array[-1]
dig_array[-1].reshape(-1,1)
runfile('C:/Users/user/.spyder-py3/SVM.py', wdir='C:/Users/user/.spyder-py3')
digits.data[-1]
digits.data[879]
type(digits.data[-1])
type(digits.data[879])
help(clf.predict)
runfile('C:/Users/user/.spyder-py3/SVMCreditCard.py', wdir='C:/Users/user/.spyder-py3')
file_obj.columns
len(file_obj.columns)
file_obj['Class'].value_counts()
file_obj['Class'].count()
runfile('C:/Users/user/.spyder-py3/SVMCreditCard.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/IrisWork.py', wdir='C:/Users/user/.spyder-py3')

## ---(Fri Apr 13 12:07:53 2018)---
runfile('C:/Users/user/.spyder-py3/IrisWork.py', wdir='C:/Users/user/.spyder-py3')
clf
svm.clf
SVC.clf
runfile('C:/Users/user/.spyder-py3/IrisWork.py', wdir='C:/Users/user/.spyder-py3')
array
len(array)
runfile('C:/Users/user/.spyder-py3/IrisWork.py', wdir='C:/Users/user/.spyder-py3')
x_min
x_max
X
from sklearn import datasets
help(datasets)
iris_dataset['data']
iris_dataset['target']
type(iris)
type(iris.data)
iris.data
iris.shape
iris.shape()
iris.data.shape
iris.data[:,0:2]
iris.data[:,:2]
iris.data[:,:2].shape
help(scatter)
help(plt.scatter)
def visuvalize_sepal_data():
	iris = datasets.load_iris()
	X = iris.data[:, :2]  # we only take the first two features.
	y = iris.target
	plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm)
	plt.xlabel('Sepal length')
	plt.ylabel('Sepal width')
	plt.title('Sepal Width & Length')
	plt.show()

visuvalize_sepal_data()
runfile('C:/Users/user/.spyder-py3/Naive_Spam.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/dict_kMeans.py', wdir='C:/Users/user/.spyder-py3')
list(dict1)
runfile('C:/Users/user/.spyder-py3/dict_kMeans.py', wdir='C:/Users/user/.spyder-py3')
type(random.sample(list(dict1),2))
runfile('C:/Users/user/.spyder-py3/dict_kMeans.py', wdir='C:/Users/user/.spyder-py3')
runfile('C:/Users/user/.spyder-py3/untitled2.py', wdir='C:/Users/user/.spyder-py3')
file_obj.columns
runfile('C:/Users/user/.spyder-py3/untitled2.py', wdir='C:/Users/user/.spyder-py3')
import pandas as pd
import numpy as np
file_obj=pd.read_csv("adult.data.csv")
print(file_obj)
print(file_obj.dtypes)
print(file_obj.columns)
file_obj.replace(" ?",np.NaN,inplace=True)
print(file_obj["workclass"])
file_obj["age"].fillna(file_obj["age"].mean())
print(file_obj["age"].value_counts(dropna=False))
runfile('C:/Users/user/.spyder-py3/untitled2.py', wdir='C:/Users/user/.spyder-py3')
file_obj.columns
file_obj["workclass"].value_counts(dropna=False)
runfile('C:/Users/user/.spyder-py3/untitled2.py', wdir='C:/Users/user/.spyder-py3')
file_obj["workclass"].value_counts(dropna=False)
runfile('C:/Users/user/.spyder-py3/untitled2.py', wdir='C:/Users/user/.spyder-py3')
file_obj["workclass"].value_counts(dropna=False)
file_obj["workclass"].mode()
help(score)
help(KMeans.score)
runfile('C:/Users/user/.spyder-py3/untitled2.py', wdir='C:/Users/user/.spyder-py3')
kmeans.predict(x_test)
runfile('C:/Users/user/.spyder-py3/untitled2.py', wdir='C:/Users/user/.spyder-py3')
y
runfile('C:/Users/user/.spyder-py3/untitled2.py', wdir='C:/Users/user/.spyder-py3')
list(x_train)
runfile('C:/Users/user/.spyder-py3/IrisForAll.py', wdir='C:/Users/user/.spyder-py3')
lr.coef_
lr.coef_[0]
lr.coef_[0][0]
lr.coef_[1]
lr.coef_
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
lr=LinearRegression()
lr.fit(x_train,y_train)
score_value=lr.score(x_test,y_test)
print(score_value)
#sepal_length sepal_width petal_length petal_width
print(lr.coef_)
print(lr.intercept_)
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
lr=LinearRegression()
lr.fit(x_train,y_train)
print(lr.predict(x_test,y_test))
score_value=lr.score(x_test,y_test)
print(score_value)
#sepal_length sepal_width petal_length petal_width
print(lr.coef_)
print(lr.intercept_)
print(lr.predict(x_test,y_test))
print(lr.predict(x_test))
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
lr=LinearRegression()
lr.fit(x_train,y_train)
predict_val=lr.predict(x_test)
score_value=lr.score(x_test,y_test)
print(score_value)
#sepal_length sepal_width petal_length petal_width
print(lr.coef_)
print(lr.intercept_)
for i in range(0,len(y_test)):
    print("%d vs %d"%(np.array(y_test[i]),predict_val[i]))

for i in range(0,len(y_test)):
    print("%d vs %d"%(np.array(y_test[i]),predict_val[i]))

from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
lr=LinearRegression()
lr.fit(x_train,y_train)
predict_val=lr.predict(x_test)
score_value=lr.score(x_test,y_test)
print(score_value)
#sepal_length sepal_width petal_length petal_width
print(lr.coef_)
print(lr.intercept_)
for i in range(0,len(y_test)):
    print("%d vs %d"%(np.array(y_test)[i],predict_val[i]))

from sklearn import datasets
from sklearn.linear_model import LogisticRegression 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
lr=LogisticRegression()
lr.fit(x_train,y_train)
lr.predict(x_test)
score_value=lr.score(x_test,y_test)
print(score_value)
from sklearn import datasets
from sklearn.linear_model import LogisticRegression 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
lr=LogisticRegression()
lr.fit(x_train,y_train)
pr=lr.predict(x_test)
score_value=lr.score(x_test,y_test)
print(score_value)
print(accuracy_score(y_test,pr))
from sklearn import datasets
from sklearn.linear_model import LogisticRegression 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
lr=LogisticRegression()
lr.fit(x_train,y_train)
pr=lr.predict(x_test)
score_value=lr.score(x_test,y_test)
print(score_value)
print(accuracy_score(y_test,pr))
for i in range(0,len(y_test)):
     print("%d vs %d"%(np.array(y_test)[i],pr[i]))

from sklearn import datasets
from sklearn.linear_model import LogisticRegression 
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np

iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
lr=LogisticRegression()
lr.fit(x_train,y_train)
pr=lr.predict(x_test)
score_value=lr.score(x_test,y_test)
print(score_value)
print(accuracy_score(y_test,pr))
for i in range(0,len(y_test)):
     print("%d vs %d"%(np.array(y_test)[i],pr[i]))

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import svm
iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
clf=svm.SVC()
clf.fit(x_train,y_train)
pr=clf.predict(x_test)
print(accuracy_score(y_test,pr))
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
from sklearn import svm
iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
clf=svm.SVC()
clf.fit(x_train,y_train)
pr=clf.predict(x_test)
print(accuracy_score(y_test,pr))
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import svm
iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
clf=svm.SVC()
clf.fit(x_train,y_train)
pr=clf.predict(x_test)
print(accuracy_score(y_test,pr))
for i in range(0,len(y_test)):
      print("%d vs %d"%(np.array(y_test)[i],pr[i]))

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import svm
iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=8)
clf=svm.SVC()
clf.fit(x_train,y_train)
pr=clf.predict(x_test)
print(accuracy_score(y_test,pr))
for i in range(0,len(y_test)):
      print("%d vs %d"%(np.array(y_test)[i],pr[i]))

from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
pr=nb.predict(y_test)
print(accuracy_score(y_test,pr))
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import svm
iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=8)
"""clf=svm.SVC()
clf.fit(x_train,y_train)
pr=clf.predict(x_test)
print(accuracy_score(y_test,pr))
for i in range(0,len(y_test)):
      print("%d vs %d"%(np.array(y_test)[i],pr[i]))

"""
#====================================
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
pr=nb.predict(y_test)
print(accuracy_score(y_test,pr))
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import svm
iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=8)
#====================================
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
pr=nb.predict(y_test)
print(accuracy_score(y_test,pr))
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import svm
iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
#====================================
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
pr=nb.predict(y_test)
print(accuracy_score(y_test,pr))
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import svm
iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)
#====================================
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
pr=nb.predict(y_test)
print(nb.score(x_test,y_test))
print(accuracy_score(y_test,pr))
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import svm
iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=8)
#====================================
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
pr=nb.predict(y_test)
print(nb.score(x_test,y_test))
print(accuracy_score(y_test,pr))
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
from sklearn import svm
iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=8)
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
pr=nb.predict(y_test)
print(nb.score(x_test,y_test))
print(accuracy_score(y_test,pr))
from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=8)
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
pr=nb.predict(y_test)
print(nb.score(x_test,y_test))
print(accuracy_score(y_test,pr))
for i in range(0,len(y_test)):
      print("%d vs %d"%(np.array(y_test)[i],pr[i]))

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=8)
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
pr=nb.predict(y_test)
print(nb.score(x_test,y_test))
print(accuracy_score(y_test,pr))
for i in range(0,len(y_test)):
      print("%f vs %f"%(np.array(y_test)[i],pr[i]))

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=8)
from sklearn.naive_bayes import GaussianNB
nb=GaussianNB()
nb.fit(x_train,y_train)
pr=nb.predict(y_test)
print(nb.score(x_test,y_test))
print(accuracy_score(y_test,pr))
for i in range(0,len(y_test)):
      print("%f vs %f"%(np.array(y_test)[i],pr[i]))

from sklearn import datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
iris_dataset=datasets.load_iris()
x=pd.DataFrame(iris_dataset.data)
y=pd.DataFrame(iris_dataset.target)
x.columns=iris_dataset.feature_names
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=8)
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
km=KMeans(n_cluster=2)
km.fit(x_train)
pr=km.predict(x_test)
print(accuracy_score(y_test,pr))
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
iris_dataset.data
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import seaborn as sb

iris_dataset = datasets.load_iris()
x = pd.DataFrame(iris_dataset.data)
y = pd.DataFrame(iris_dataset.target)
x.columns = iris_dataset.feature_names
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
lr = LinearRegression()
lr.fit(x_train, y_train)
predict_val = lr.predict(x_test)
score_value = lr.score(x_test, y_test)
#scc=accuracy_score(y_test, predict_val)
print(score_value)
print(accuracy_score(y_test, predict_val))
y_test
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import seaborn as sb

iris_dataset = datasets.load_iris()
x = pd.DataFrame(iris_dataset.data)
y = pd.DataFrame(iris_dataset.target)
x.columns = iris_dataset.feature_names
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
lr = LinearRegression()
lr.fit(x_train, y_train)
predict_val = lr.predict(x_test)
score_value = lr.score(x_test, y_test)
#scc=accuracy_score(y_test, predict_val)
print(score_value)
print(accuracy_score(np.array(y_test), predict_val))
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import seaborn as sb

iris_dataset = datasets.load_iris()
x = pd.DataFrame(iris_dataset.data)
y = pd.DataFrame(iris_dataset.target)
x.columns = iris_dataset.feature_names
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
lr = LinearRegression()
lr.fit(x_train, y_train)
predict_val = lr.predict(x_test)
score_value = lr.score(x_test, y_test)
#scc=accuracy_score(y_test, predict_val)
print(score_value)
print(accuracy_score(np.array(y_test[0]), predict_val))
np.array(y_test)
y_test[0]
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import seaborn as sb

iris_dataset = datasets.load_iris()
x = pd.DataFrame(iris_dataset.data)
y = pd.DataFrame(iris_dataset.target)
x.columns = iris_dataset.feature_names
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
lr = LinearRegression()
lr.fit(x_train, y_train)
predict_val = lr.predict(x_test)
score_value = lr.score(x_test, y_test)
#scc=accuracy_score(y_test, predict_val)
print(score_value)
print(accuracy_score(y_test[0], predict_val))
type(y_test)
type(predict_val)
y_test[0]
type(y_test[0])
y_test[0]
np.array(y_test)
y_test[0]
list(y_test[0])
np.array(list(y_test[0]))
predict_val
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import seaborn as sb

iris_dataset = datasets.load_iris()
x = pd.DataFrame(iris_dataset.data)
y = pd.DataFrame(iris_dataset.target)
x.columns = iris_dataset.feature_names
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
lr = LinearRegression()
lr.fit(x_train, y_train)
predict_val = lr.predict(x_test)
score_value = lr.score(x_test, y_test)
#scc=accuracy_score(y_test, predict_val)
print(score_value)
print(accuracy_score(np.array(list(y_test[0])), predict_val))
predict_val
list(predict_val)
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import seaborn as sb

iris_dataset = datasets.load_iris()
x = pd.DataFrame(iris_dataset.data)
y = pd.DataFrame(iris_dataset.target)
x.columns = iris_dataset.feature_names
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
lr = LinearRegression()
lr.fit(x_train, y_train)
predict_val = lr.predict(x_test)
score_value = lr.score(x_test, y_test)
#scc=accuracy_score(y_test, predict_val)
import itertools
merged = list(itertools.chain.from_iterable(predict_val))
print(score_value)
print(accuracy_score(np.array(list(y_test[0])), predict_val))
merged
from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
import pandas as pd
import numpy as np
import seaborn as sb

iris_dataset = datasets.load_iris()
x = pd.DataFrame(iris_dataset.data)
y = pd.DataFrame(iris_dataset.target)
x.columns = iris_dataset.feature_names
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
lr = LinearRegression()
lr.fit(x_train, y_train)
predict_val = lr.predict(x_test)
score_value = lr.score(x_test, y_test)
#scc=accuracy_score(y_test, predict_val)
import itertools
merged = list(itertools.chain.from_iterable(predict_val))
print(score_value)
print(accuracy_score(np.array(list(y_test[0])), merged))
runfile('C:/Users/user/.spyder-py3/KmeansImplement.py', wdir='C:/Users/user/.spyder-py3')
import numpy as np
import pandas as pd

##X = np.array([[1, 2],
##              [1.5, 1.8],
##              [5, 8],
##              [8, 8],
##              [1, 0.6],
##              [9, 11]])
##
##
##colors = ['r','g','b','c','k','o','y']



class K_Means:
    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
    
    def fit(self,data):
        
        self.centroids = {}
        
        for i in range(self.k):
            self.centroids[i] = data[i]
        
        for i in range(self.max_iter):
            self.classifications = {}
            
            for i in range(self.k):
                self.classifications[i] = []
            
            for featureset in X:
                distances = [np.linalg.norm(featureset-self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)
            
            prev_centroids = dict(self.centroids)
            
            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification],axis=0)
            
            optimized = True
            
            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False
            
            if optimized:
                break
    
    def predict(self,data):
        distances = [np.linalg.norm(data-self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification



# https://pythonprogramming.net/static/downloads/machine-learning-data/titanic.xls
df = pd.read_excel('titanic.xls')
df.drop(['body','name'], 1, inplace=True)
#df.convert_objects(convert_numeric=True)
print(df.head())
df.fillna(0,inplace=True)

def handle_non_numerical_data(df):
    
    # handling non-numerical data: must convert.
    columns = df.columns.values
    
    for column in columns:
        text_digit_vals = {}
        def convert_to_int(val):
            return text_digit_vals[val]
        
        #print(column,df[column].dtype)
        if df[column].dtype != np.int64 and df[column].dtype != np.float64:
            
            column_contents = df[column].values.tolist()
            #finding just the uniques
            unique_elements = set(column_contents)
            # great, found them. 
            x = 0
            for unique in unique_elements:
                if unique not in text_digit_vals:
                    # creating dict that contains new
                    # id per unique string
                    text_digit_vals[unique] = x
                    x+=1
            # now we map the new "id" vlaue
            # to replace the string. 
            df[column] = list(map(convert_to_int,df[column]))
    
    return df


df = handle_non_numerical_data(df)
print(df.head())

# add/remove features just to see impact they have.
df.drop(['ticket','home.dest'], 1, inplace=True)


X = np.array(df.drop(['survived'], 1).astype(float))
X = preprocessing.scale(X)
y = np.array(df['survived'])

#X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.5)

clf = K_Means()
clf.fit(X)

correct = 0
for i in range(len(X)):
    
    predict_me = np.array(X[i].astype(float))
    predict_me = predict_me.reshape(-1, len(predict_me))
    prediction = clf.predict(predict_me)
    if prediction == y[i]:
        correct += 1



print(correct/len(X))
runfile('C:/Users/user/.spyder-py3/SirajRaval.py', wdir='C:/Users/user/.spyder-py3')
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def load_dataset(name):
    return np.loadtxt(name)



def euclidian(a, b):
    return np.linalg.norm(a-b)



def plot(dataset, history_centroids, belongs_to):
    colors = ['r', 'g']
    
    fig, ax = plt.subplots()
    
    for index in range(dataset.shape[0]):
        instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
        for instance_index in instances_close:
            ax.plot(dataset[instance_index][0], dataset[instance_index][1], (colors[index] + 'o'))
    
    history_points = []
    for index, centroids in enumerate(history_centroids):
        for inner, item in enumerate(centroids):
            if index == 0:
                history_points.append(ax.plot(item[0], item[1], 'bo')[0])
            else:
                history_points[inner].set_data(item[0], item[1])
                print("centroids {} {}".format(index, item))
                
                plt.pause(0.8)



def kmeans(k, epsilon=0, distance='euclidian'):
    history_centroids = []
    if distance == 'euclidian':
        dist_method = euclidian
    dataset = load_dataset('durudataset.txt')
    # dataset = dataset[:, 0:dataset.shape[1] - 1]
    num_instances, num_features = dataset.shape
    prototypes = dataset[np.random.randint(0, num_instances - 1, size=k)]
    history_centroids.append(prototypes)
    prototypes_old = np.zeros(prototypes.shape)
    belongs_to = np.zeros((num_instances, 1))
    norm = dist_method(prototypes, prototypes_old)
    iteration = 0
    while norm > epsilon:
        iteration += 1
        norm = dist_method(prototypes, prototypes_old)
        prototypes_old = prototypes
        for index_instance, instance in enumerate(dataset):
            dist_vec = np.zeros((k, 1))
            for index_prototype, prototype in enumerate(prototypes):
                dist_vec[index_prototype] = dist_method(prototype,
                                                        instance)
            
            belongs_to[index_instance, 0] = np.argmin(dist_vec)
        
        tmp_prototypes = np.zeros((k, num_features))
        
        for index in range(len(prototypes)):
            instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
            prototype = np.mean(dataset[instances_close], axis=0)
            # prototype = dataset[np.random.randint(0, num_instances, size=1)[0]]
            tmp_prototypes[index, :] = prototype
        
        prototypes = tmp_prototypes
        
        history_centroids.append(tmp_prototypes)
    
    # plot(dataset, history_centroids, belongs_to)
    
    return prototypes, history_centroids, belongs_to



def execute():
    dataset = load_dataset('durudataset.txt')
    centroids, history_centroids, belongs_to = kmeans(2)
    plot(dataset, history_centroids, belongs_to)


execute()
dataset.shape
dataset = load_dataset('durudataset.txt')
dataset.shape
prototypes = dataset[np.random.randint(0, num_instances - 1, size=k)]
num_instances, num_features = dataset.shape
num_instances
prototypes = dataset[np.random.randint(0, num_instances - 1, size=k)]
prototypes = dataset[np.random.randint(0, num_instances - 1, size=2)]
prototypes
history_centroids = []
prototypes_old = np.zeros(prototypes.shape)
prototypes_old
belongs_to = np.zeros((num_instances, 1))
belongs_to
norm = dist_method(prototypes, prototypes_old)
prototypes
prototypes_old
norm = dist_method(prototypes, prototypes_old)
norm = euclidian(prototypes, prototypes_old)
norm
runfile('C:/Users/user/.spyder-py3/SirajRaval.py', wdir='C:/Users/user/.spyder-py3')
def kmeans(k, epsilon=0, distance='euclidian'):
    history_centroids = []
    if distance == 'euclidian':
        dist_method = euclidian
    dataset = load_dataset('durudataset.txt')
    # dataset = dataset[:, 0:dataset.shape[1] - 1]
    num_instances, num_features = dataset.shape
    prototypes = dataset[np.random.randint(0, num_instances - 1, size=k)]
    history_centroids.append(prototypes)
    prototypes_old = np.zeros(prototypes.shape)
    belongs_to = np.zeros((num_instances, 1))
    norm = dist_method(prototypes, prototypes_old)
    iteration = 0
    while norm > epsilon:
        iteration += 1
        norm = dist_method(prototypes, prototypes_old)
        prototypes_old = prototypes
        for index_instance, instance in enumerate(dataset):
            dist_vec = np.zeros((k, 1))
            for index_prototype, prototype in enumerate(prototypes):
                dist_vec[index_prototype] = dist_method(prototype,
                                                        instance)
            
            belongs_to[index_instance, 0] = np.argmin(dist_vec)
        
        tmp_prototypes = np.zeros((k, num_features))
        
        for index in range(len(prototypes)):
            instances_close = [i for i in range(len(belongs_to)) if belongs_to[i] == index]
            prototype = np.mean(dataset[instances_close], axis=0)
            # prototype = dataset[np.random.randint(0, num_instances, size=1)[0]]
            tmp_prototypes[index, :] = prototype
        
        prototypes = tmp_prototypes
        
        history_centroids.append(tmp_prototypes)
    
    # plot(dataset, history_centroids, belongs_to)
    
    return prototypes, history_centroids, belongs_to

def euclidean_dist(x_list,y_list,cent_1,cent_2):
     math.sqrt(math.pow((x_list,cent_1),2)+math.pow((y_list,cent_2),2))

import math
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
cluster_1=[]
cluster_2=[]
centroid_for_cluster1=feature_dict[1]
centroid_for_cluster2=feature_dict[4]

def euclidean_dist(x_list,y_list,cent_1,cent_2):
     math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))


while(True):
    sum_value_x=0
    sum_value_y=0
    prev_cluster_1=cluster_1
    prev_cluster_2=cluster_2
    for ele,value in feature_dict.items():
        element=feature_dict[ele]
        check_dist_cent_1=euclidean_dist(element[0],element[1],centroid_for_cluster1[0],centroid_for_cluster1[1])
        check_dist_cent_2=euclidean_dist(element[0],element[1],centroid_for_cluster2[0],centroid_for_cluster2[1])
        if check_dist_cent_1<check_dist_cent_2 and ele not in cluster_1 :
            cluster_1.append(ele)
        elif check_dist_cent_2<check_dist_cent_1 and ele not in cluster_2:
             cluster_2.append(ele)
        elif check_dist_cent_1<check_dist_cent_2 and (ele not in cluster_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        else :
            print("Please check you r program once")
    if cluster_1==prev_cluster_1 and prev_cluster_2==cluster_2:
        break
    
    for i in cluster_1:
        element=feature_dict[i]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
    
    centroid_for_cluster1=list(sum_value_x,sum_value_y)  
    
    sum_value_x=0
    sum_value_y=0
    
    for i in cluster_2:
        element=feature_dict[i]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
    
    centroid_for_cluster2=list(sum_value_x,sum_value_y)       

while(True):
    sum_value_x=0
    sum_value_y=0
    prev_cluster_1=cluster_1
    prev_cluster_2=cluster_2
    for ele,value in feature_dict.items():
        element=feature_dict[ele]

for ele,value in feature_dict.items():
    element=feature_dict[ele]

import math
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
cluster_1=[]
cluster_2=[]
centroid_for_cluster1=feature_dict[1]
centroid_for_cluster2=feature_dict[4]

def euclidean_dist(x_list,y_list,cent_1,cent_2):
     math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))

sum_value_x=0
sum_value_y=0
prev_cluster_1=cluster_1
prev_cluster_2=cluster_2
for ele,value in feature_dict.items():
    element=feature_dict[ele]

for ele,value in feature_dict.items():
    element=feature_dict[ele]
    print("The key is",ele)
    print("The element is"element)

for ele,value in feature_dict.items():
    element=feature_dict[ele]
    print("The key is",ele)
    print("The element is",element)

check_dist_cent_1=euclidean_dist(element[0],element[1],centroid_for_cluster1[0],centroid_for_cluster1[1])
check_dist_cent_1
check_dist_cent_1=euclidean_dist(element[0],element[1],centroid_for_cluster1[0],centroid_for_cluster1[1])
check_dist_cent_2=euclidean_dist(element[0],element[1],centroid_for_cluster2[0],centroid_for_cluster2[1])
check_dist_cent_1=euclidean_dist(element[0],element[1],centroid_for_cluster1[0],centroid_for_cluster1[1])
print(check_dist_cent_1)
check_dist_cent_2=euclidean_dist(element[0],element[1],centroid_for_cluster2[0],centroid_for_cluster2[1])
print(check_dist_cent_2)
import math
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
cluster_1=[]
cluster_2=[]
centroid_for_cluster1=feature_dict[1]
centroid_for_cluster2=feature_dict[4]

def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))

sum_value_x=0
sum_value_y=0
prev_cluster_1=cluster_1
prev_cluster_2=cluster_2
for ele,value in feature_dict.items():
    element=feature_dict[ele]
    print("The key is",ele)
    print("The element is",element)
    check_dist_cent_1=euclidean_dist(element[0],element[1],centroid_for_cluster1[0],centroid_for_cluster1[1])
    print(check_dist_cent_1)
    check_dist_cent_2=euclidean_dist(element[0],element[1],centroid_for_cluster2[0],centroid_for_cluster2[1])
    print(check_dist_cent_2)

if (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) :
    cluster_1.append(ele)
elif (check_dist_cent_2<check_dist_cent_1) and (ele not in cluster_2):
     cluster_2.append(ele)
elif (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) and (ele not in cluster_2):
     cluster_2.append(ele)
else :
    print("Please check you r program once")

if cluster_1==prev_cluster_1 and prev_cluster_2==cluster_2:
    break

if (cluster_1==prev_cluster_1) and (prev_cluster_2==cluster_2):
    break

if (cluster_1==prev_cluster_1) and (prev_cluster_2==cluster_2):
    break

if (cluster_1==prev_cluster_1) and (prev_cluster_2==cluster_2):
    break

if(cluster_1==prev_cluster_1) and (prev_cluster_2==cluster_2):
    break

if((cluster_1==prev_cluster_1) and (prev_cluster_2==cluster_2)):
    break

if(() and (prev_cluster_2==cluster_2)):
    break

if cluster_1==prev_cluster_1 and cluster_2==prev_cluster_1:
    break

if cluster_1==prev_cluster_1 and cluster_2==prev_cluster_1:
     temp_count=False

for i in cluster_1:
    element=feature_dict[i]
    sum_value_x=sum_value_x+element[0]
    sum_value_y=sum_value_y+element[1]

centroid_for_cluster1=list(sum_value_x,sum_value_y)  
centroid_for_cluster1
centroid_for_cluster1=list([sum_value_x,sum_value_y])  
centroid_for_cluster1
for i in cluster_1:
    element=feature_dict[i]
    sum_value_x=sum_value_x+element[0]
    sum_value_y=sum_value_y+element[1]

centroid_for_cluster1=list([sum_value_x,sum_value_y])
centroid_for_cluster1
for i in cluster_1:
    print("key",i)
    element=feature_dict[i]
    print("Element",element)
    sum_value_x=sum_value_x+element[0]
    print(sum_value_x)
    sum_value_y=sum_value_y+element[1]
    print(sum_value_y)

import math
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
cluster_1=[]
cluster_2=[]
centroid_for_cluster1=feature_dict[1]
centroid_for_cluster2=feature_dict[4]

def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))

sum_value_x=0
sum_value_y=0
prev_cluster_1=cluster_1
prev_cluster_2=cluster_2
for ele,value in feature_dict.items():
    element=feature_dict[ele]
    print("The key is",ele)
    print("The element is",element)
    check_dist_cent_1=euclidean_dist(element[0],element[1],centroid_for_cluster1[0],centroid_for_cluster1[1])
    print(check_dist_cent_1)
    check_dist_cent_2=euclidean_dist(element[0],element[1],centroid_for_cluster2[0],centroid_for_cluster2[1])
    print(check_dist_cent_2)
    if (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) :
        cluster_1.append(ele)
    elif (check_dist_cent_2<check_dist_cent_1) and (ele not in cluster_2):
         cluster_2.append(ele)
    elif (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) and (ele not in cluster_2):
         cluster_2.append(ele)
    else :
        print("Please check you r program once")

print(cluster_1)
print(cluster_2)
for i in cluster_1:
    print("key",i)
    element=feature_dict[i]
    print("Element",element)
    sum_value_x=sum_value_x+element[0]
    print(sum_value_x)
    sum_value_y=sum_value_y+element[1]
    print(sum_value_y)

for i in cluster_1:
    print("key",i)
    element=feature_dict[i]
    print("Element",element)
    sum_value_x=sum_value_x+element[0]
    print(sum_value_x)
    sum_value_y=sum_value_y+element[1]
    print(sum_value_y)

print("Final Value",sum_value_x)
print("Final Value",sum_value_y)
for i in cluster_1:
    print("key",i)
    element=feature_dict[i]
    print("Element",element)
    sum_value_x=sum_value_x+element[0]
    print(sum_value_x)
    sum_value_y=sum_value_y+element[1]
    print(sum_value_y)

print("Final Value",sum_value_x/len(sum_value_x))
print("Final Value",sum_value_y/len(sum_value_y))
for i in cluster_1:
    print("key",i)
    element=feature_dict[i]
    print("Element",element)
    sum_value_x=sum_value_x+element[0]
    print(sum_value_x)
    sum_value_y=sum_value_y+element[1]
    print(sum_value_y)

print("Final Value",sum_value_x/len(int(cluster_1)))
print("Final Value",sum_value_y/len(int(cluster_1)))
for i in cluster_1:
    print("key",i)
    element=feature_dict[i]
    print("Element",element)
    sum_value_x=sum_value_x+element[0]
    print(sum_value_x)
    sum_value_y=sum_value_y+element[1]
    print(sum_value_y)

print("Final Value",sum_value_x/(len(cluster_1)))
print("Final Value",sum_value_y/(len(cluster_1)))
sum_value_x=0
sum_value_y=0
for i in cluster_1:
    print("key",i)
    element=feature_dict[i]
    print("Element",element)
    sum_value_x=sum_value_x+element[0]
    print(sum_value_x)
    sum_value_y=sum_value_y+element[1]
    print(sum_value_y)

print("Final Value",sum_value_x/(len(cluster_1)))
print("Final Value",sum_value_y/(len(cluster_1)))
centroid_for_cluster1=list([sum_value_x,sum_value_y])  
centroid_for_cluster1=list([sum_value_x,sum_value_y])  
centroid_for_cluster1
centroid_for_cluster1=list([sum_value_x/(len(cluster_1)),sum_value_y/(len(cluster_1))])  
sum_value_x=0
sum_value_y=0
for i in cluster_2:
    element=feature_dict[i]
    sum_value_x=sum_value_x+element[0]
    sum_value_y=sum_value_y+element[1]


print("Final Value",sum_value_x/(len(cluster_2)))
print("Final Value",sum_value_y/(len(cluster_2)))
centroid_for_cluster1=list([sum_value_x/(len(cluster_2)),sum_value_y/(len(cluster_2))])
import math
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
cluster_1=[]
cluster_2=[]
centroid_for_cluster1=feature_dict[1]
centroid_for_cluster2=feature_dict[4]

def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))

temp_count=True
while temp_count:
    sum_value_x=0
    sum_value_y=0
    prev_cluster_1=cluster_1
    prev_cluster_2=cluster_2
    
    for ele,value in feature_dict.items():
        element=feature_dict[ele]
        print("The key is",ele)
        print("The element is",element)
        check_dist_cent_1=euclidean_dist(element[0],element[1],centroid_for_cluster1[0],centroid_for_cluster1[1])
        print(check_dist_cent_1)
        check_dist_cent_2=euclidean_dist(element[0],element[1],centroid_for_cluster2[0],centroid_for_cluster2[1])
        print(check_dist_cent_2)
        if (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) :
            cluster_1.append(ele)
        elif (check_dist_cent_2<check_dist_cent_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        elif (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        else :
            print("Please check you r program once")
    print(cluster_1)
    print(cluster_2)
    if cluster_1==prev_cluster_1 and cluster_2==prev_cluster_1:
         temp_count=False
    
    for i in cluster_1:
        print("key",i)
        element=feature_dict[i]
        print("Element",element)
        sum_value_x=sum_value_x+element[0]
        print(sum_value_x)
        sum_value_y=sum_value_y+element[1]
        print(sum_value_y)
    print("Final Value",sum_value_x/(len(cluster_1)))
    print("Final Value",sum_value_y/(len(cluster_1)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_1)),sum_value_y/(len(cluster_1))])  
    
    sum_value_x=0
    sum_value_y=0
    
    for i in cluster_2:
        element=feature_dict[i]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
    
    print("Final Value",sum_value_x/(len(cluster_2)))
    print("Final Value",sum_value_y/(len(cluster_2)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_2)),sum_value_y/(len(cluster_2))])  


lis1=[1,2,3]
lis=[1,2,3]
lis1==lis
import math
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
cluster_1=[]
cluster_2=[]
centroid_for_cluster1=feature_dict[1]
centroid_for_cluster2=feature_dict[4]

def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))

temp_count=True
count=0
while temp_count or count<4:
    sum_value_x=0
    sum_value_y=0
    prev_cluster_1=cluster_1
    prev_cluster_2=cluster_2
    
    for ele,value in feature_dict.items():
        element=feature_dict[ele]
        print("The key is",ele)
        print("The element is",element)
        check_dist_cent_1=euclidean_dist(element[0],element[1],centroid_for_cluster1[0],centroid_for_cluster1[1])
        print(check_dist_cent_1)
        check_dist_cent_2=euclidean_dist(element[0],element[1],centroid_for_cluster2[0],centroid_for_cluster2[1])
        print(check_dist_cent_2)
        if (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) :
            cluster_1.append(ele)
        elif (check_dist_cent_2<check_dist_cent_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        elif (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        else :
            print("Please check you r program once")
    print(cluster_1)
    print(cluster_2)
    if cluster_1==prev_cluster_1 and cluster_2==prev_cluster_1:
         temp_count=False
    
    for i in cluster_1:
        print("key",i)
        element=feature_dict[i]
        print("Element",element)
        sum_value_x=sum_value_x+element[0]
        print(sum_value_x)
        sum_value_y=sum_value_y+element[1]
        print(sum_value_y)
    print("Final Value",sum_value_x/(len(cluster_1)))
    print("Final Value",sum_value_y/(len(cluster_1)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_1)),sum_value_y/(len(cluster_1))])  
    
    sum_value_x=0
    sum_value_y=0
    
    for i in cluster_2:
        element=feature_dict[i]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
    
    print("Final Value",sum_value_x/(len(cluster_2)))
    print("Final Value",sum_value_y/(len(cluster_2)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_2)),sum_value_y/(len(cluster_2))])  
    ++count


import math
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
cluster_1=[]
cluster_2=[]
centroid_for_cluster1=feature_dict[1]
centroid_for_cluster2=feature_dict[4]

def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))

temp_count=True
count=0
while temp_count or count<4:
    sum_value_x=0
    sum_value_y=0
    prev_cluster_1=cluster_1
    prev_cluster_2=cluster_2
    
    for ele,value in feature_dict.items():
        element=feature_dict[ele]
        print("The key is",ele)
        print("The element is",element)
        check_dist_cent_1=euclidean_dist(element[0],element[1],centroid_for_cluster1[0],centroid_for_cluster1[1])
        print(check_dist_cent_1)
        check_dist_cent_2=euclidean_dist(element[0],element[1],centroid_for_cluster2[0],centroid_for_cluster2[1])
        print(check_dist_cent_2)
        if (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) :
            cluster_1.append(ele)
        elif (check_dist_cent_2<check_dist_cent_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        elif (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        else :
            print("Please check you r program once")
    print(cluster_1)
    print(cluster_2)
    if cluster_1==prev_cluster_1 and cluster_2==prev_cluster_1:
         temp_count=False
    count=count+1
    for i in cluster_1:
        print("key",i)
        element=feature_dict[i]
        print("Element",element)
        sum_value_x=sum_value_x+element[0]
        print(sum_value_x)
        sum_value_y=sum_value_y+element[1]
        print(sum_value_y)
    print("Final Value",sum_value_x/(len(cluster_1)))
    print("Final Value",sum_value_y/(len(cluster_1)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_1)),sum_value_y/(len(cluster_1))])  
    
    sum_value_x=0
    sum_value_y=0
    
    for i in cluster_2:
        element=feature_dict[i]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
    
    print("Final Value",sum_value_x/(len(cluster_2)))
    print("Final Value",sum_value_y/(len(cluster_2)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_2)),sum_value_y/(len(cluster_2))])  



import math
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
cluster_1=[]
cluster_2=[]
centroid_for_cluster1=feature_dict[1]
centroid_for_cluster2=feature_dict[4]
def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))

import math
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
cluster_1=[]
cluster_2=[]
centroid_for_cluster1=feature_dict[1]
centroid_for_cluster2=feature_dict[4]

def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))

temp_count=True
count=0
while temp_count or count<4:
    print("Inside While============================================================")
    sum_value_x=0
    sum_value_y=0
    prev_cluster_1=cluster_1
    prev_cluster_2=cluster_2
    
    for ele,value in feature_dict.items():
        print("Inside first for ==============================================")
        element=feature_dict[ele]
        print("The key is",ele)
        print("The element is",element)
        check_dist_cent_1=euclidean_dist(element[0],element[1],centroid_for_cluster1[0],centroid_for_cluster1[1])
        print(check_dist_cent_1)
        check_dist_cent_2=euclidean_dist(element[0],element[1],centroid_for_cluster2[0],centroid_for_cluster2[1])
        print(check_dist_cent_2)
        if (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) :
            cluster_1.append(ele)
        elif (check_dist_cent_2<check_dist_cent_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        elif (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        else :
            print("Please check you r program once")
    print(cluster_1)
    print(cluster_2)
    if cluster_1==prev_cluster_1 and cluster_2==prev_cluster_1:
         temp_count=False
    count=count+1
    for i in cluster_1:
        print("Inside second for ==============================================")
        print("key",i)
        element=feature_dict[i]
        print("Element",element)
        sum_value_x=sum_value_x+element[0]
        print(sum_value_x)
        sum_value_y=sum_value_y+element[1]
        print(sum_value_y)
    print("Final Value",sum_value_x/(len(cluster_1)))
    print("Final Value",sum_value_y/(len(cluster_1)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_1)),sum_value_y/(len(cluster_1))])  
    
    sum_value_x=0
    sum_value_y=0
    
    for i in cluster_2:
        print("Inside third for ==============================================")
        element=feature_dict[i]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
    
    print("Final Value",sum_value_x/(len(cluster_2)))
    print("Final Value",sum_value_y/(len(cluster_2)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_2)),sum_value_y/(len(cluster_2))])  



import math
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
cluster_1=[]
cluster_2=[]
centroid_for_cluster1=feature_dict[1]
centroid_for_cluster2=feature_dict[4]

def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))

temp_count=True
count=0
while count<4  and temp_count :
    print("Inside While============================================================")
    sum_value_x=0
    sum_value_y=0
    prev_cluster_1=cluster_1
    prev_cluster_2=cluster_2
    
    for ele,value in feature_dict.items():
        print("Inside first for ==============================================")
        element=feature_dict[ele]
        print("The key is",ele)
        print("The element is",element)
        check_dist_cent_1=euclidean_dist(element[0],element[1],centroid_for_cluster1[0],centroid_for_cluster1[1])
        print(check_dist_cent_1)
        check_dist_cent_2=euclidean_dist(element[0],element[1],centroid_for_cluster2[0],centroid_for_cluster2[1])
        print(check_dist_cent_2)
        if (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) :
            cluster_1.append(ele)
        elif (check_dist_cent_2<check_dist_cent_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        elif (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        else :
            print("Please check you r program once")
    print(cluster_1)
    print(cluster_2)
    if cluster_1==prev_cluster_1 and cluster_2==prev_cluster_1:
         temp_count=False
    count=count+1
    for i in cluster_1:
        print("Inside second for ==============================================")
        print("key",i)
        element=feature_dict[i]
        print("Element",element)
        sum_value_x=sum_value_x+element[0]
        print(sum_value_x)
        sum_value_y=sum_value_y+element[1]
        print(sum_value_y)
    print("Final Value",sum_value_x/(len(cluster_1)))
    print("Final Value",sum_value_y/(len(cluster_1)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_1)),sum_value_y/(len(cluster_1))])  
    
    sum_value_x=0
    sum_value_y=0
    
    for i in cluster_2:
        print("Inside third for ==============================================")
        element=feature_dict[i]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
    
    print("Final Value",sum_value_x/(len(cluster_2)))
    print("Final Value",sum_value_y/(len(cluster_2)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_2)),sum_value_y/(len(cluster_2))])  



import math
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
cluster_1=[]
cluster_2=[]
centroid_for_cluster1=feature_dict[1]
centroid_for_cluster2=feature_dict[4]

def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))

temp_count=True
count=0
while count<4  and temp_count :
    print("Inside While============================================================")
    sum_value_x=0
    sum_value_y=0
    prev_cluster_1=cluster_1
    prev_cluster_2=cluster_2
    
    for ele,value in feature_dict.items():
        print("Inside first for ==============================================")
        element=feature_dict[ele]
        print("The key is",ele)
        print("The element is",element)
        check_dist_cent_1=euclidean_dist(element[0],element[1],centroid_for_cluster1[0],centroid_for_cluster1[1])
        print(check_dist_cent_1)
        check_dist_cent_2=euclidean_dist(element[0],element[1],centroid_for_cluster2[0],centroid_for_cluster2[1])
        print(check_dist_cent_2)
        if (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) :
            cluster_1.append(ele)
        elif (check_dist_cent_2<check_dist_cent_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        elif (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        else :
            print("Please check you r program once")
    print(cluster_1)
    print(cluster_2)
    count=count+1
    print("The count is ",count)
    print("===================")
    if cluster_1==prev_cluster_1 and cluster_2==prev_cluster_1:
         temp_count=False
    
    for i in cluster_1:
        print("Inside second for ==============================================")
        print("key",i)
        element=feature_dict[i]
        print("Element",element)
        sum_value_x=sum_value_x+element[0]
        print(sum_value_x)
        sum_value_y=sum_value_y+element[1]
        print(sum_value_y)
    print("Final Value",sum_value_x/(len(cluster_1)))
    print("Final Value",sum_value_y/(len(cluster_1)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_1)),sum_value_y/(len(cluster_1))])  
    
    sum_value_x=0
    sum_value_y=0
    
    for i in cluster_2:
        print("Inside third for ==============================================")
        element=feature_dict[i]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
    
    print("Final Value",sum_value_x/(len(cluster_2)))
    print("Final Value",sum_value_y/(len(cluster_2)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_2)),sum_value_y/(len(cluster_2))])  



import math
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
cluster_1=[]
cluster_2=[]
centroid_for_cluster1=feature_dict[1]
centroid_for_cluster2=feature_dict[4]

def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))

temp_count=True
count=0
while count<4  and temp_count :
    print("Inside While============================================================")
    sum_value_x=0
    sum_value_y=0
    prev_cluster_1=cluster_1
    prev_cluster_2=cluster_2
    
    for ele,value in feature_dict.items():
        print("Inside first for ==============================================")
        element=feature_dict[ele]
        print("The key is",ele)
        print("The element is",element)
        check_dist_cent_1=euclidean_dist(element[0],element[1],centroid_for_cluster1[0],centroid_for_cluster1[1])
        print(check_dist_cent_1)
        check_dist_cent_2=euclidean_dist(element[0],element[1],centroid_for_cluster2[0],centroid_for_cluster2[1])
        print(check_dist_cent_2)
        if (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) :
            cluster_1.append(ele)
        elif (check_dist_cent_2<check_dist_cent_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        elif (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        else :
            print("Please check you r program once")
    print(cluster_1)
    print(cluster_2)
    count=count+1
    print("The count is ",count)
    print("===================")
    if cluster_1==prev_cluster_1 and cluster_2==prev_cluster_2:
         temp_count=False
    
    for i in cluster_1:
        print("Inside second for ==============================================")
        print("key",i)
        element=feature_dict[i]
        print("Element",element)
        sum_value_x=sum_value_x+element[0]
        print(sum_value_x)
        sum_value_y=sum_value_y+element[1]
        print(sum_value_y)
    print("Final Value",sum_value_x/(len(cluster_1)))
    print("Final Value",sum_value_y/(len(cluster_1)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_1)),sum_value_y/(len(cluster_1))])  
    
    sum_value_x=0
    sum_value_y=0
    
    for i in cluster_2:
        print("Inside third for ==============================================")
        element=feature_dict[i]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
    
    print("Final Value",sum_value_x/(len(cluster_2)))
    print("Final Value",sum_value_y/(len(cluster_2)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_2)),sum_value_y/(len(cluster_2))])  



import math
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
cluster_1=[]
cluster_2=[]
centroid_for_cluster1=feature_dict[1]
centroid_for_cluster2=feature_dict[4]

def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))

temp_count=True

while temp_count :
    print("Inside While============================================================")
    sum_value_x=0
    sum_value_y=0
    prev_cluster_1=cluster_1
    prev_cluster_2=cluster_2
    
    for ele,value in feature_dict.items():
        print("Inside first for ==============================================")
        element=feature_dict[ele]
        print("The key is",ele)
        print("The element is",element)
        check_dist_cent_1=euclidean_dist(element[0],element[1],centroid_for_cluster1[0],centroid_for_cluster1[1])
        print(check_dist_cent_1)
        check_dist_cent_2=euclidean_dist(element[0],element[1],centroid_for_cluster2[0],centroid_for_cluster2[1])
        print(check_dist_cent_2)
        if (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) :
            cluster_1.append(ele)
        elif (check_dist_cent_2<check_dist_cent_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        elif (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        else :
            print("Please check you r program once")
    print(cluster_1)
    print(cluster_2)
    #count=count+1
    print("The count is ",count)
    print("===================")
    if cluster_1==prev_cluster_1 and cluster_2==prev_cluster_2:
         temp_count=False
    
    for i in cluster_1:
        print("Inside second for ==============================================")
        print("key",i)
        element=feature_dict[i]
        print("Element",element)
        sum_value_x=sum_value_x+element[0]
        print(sum_value_x)
        sum_value_y=sum_value_y+element[1]
        print(sum_value_y)
    print("Final Value",sum_value_x/(len(cluster_1)))
    print("Final Value",sum_value_y/(len(cluster_1)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_1)),sum_value_y/(len(cluster_1))])  
    
    sum_value_x=0
    sum_value_y=0
    
    for i in cluster_2:
        print("Inside third for ==============================================")
        element=feature_dict[i]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
    
    print("Final Value",sum_value_x/(len(cluster_2)))
    print("Final Value",sum_value_y/(len(cluster_2)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_2)),sum_value_y/(len(cluster_2))])  



import math
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
cluster_1=[]
cluster_2=[]
centroid_for_cluster1=feature_dict[1]
centroid_for_cluster2=feature_dict[4]

def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))

temp_count=True

while temp_count :
    print("Inside While============================================================")
    sum_value_x=0
    sum_value_y=0
    prev_cluster_1=cluster_1
    prev_cluster_2=cluster_2
    
    for ele,value in feature_dict.items():
        print("Inside first for ==============================================")
        element=feature_dict[ele]
        print("The key is",ele)
        print("The element is",element)
        check_dist_cent_1=euclidean_dist(element[0],element[1],centroid_for_cluster1[0],centroid_for_cluster1[1])
        print(check_dist_cent_1)
        check_dist_cent_2=euclidean_dist(element[0],element[1],centroid_for_cluster2[0],centroid_for_cluster2[1])
        print(check_dist_cent_2)
        if (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) :
            cluster_1.append(ele)
        elif (check_dist_cent_2<check_dist_cent_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        elif (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        else :
            print("Please check you r program once")
    print(cluster_1)
    print(cluster_2)
    #count=count+1
    
    if cluster_1==prev_cluster_1 and cluster_2==prev_cluster_2:
         temp_count=False
    
    for i in cluster_1:
        print("Inside second for ==============================================")
        print("key",i)
        element=feature_dict[i]
        print("Element",element)
        sum_value_x=sum_value_x+element[0]
        print(sum_value_x)
        sum_value_y=sum_value_y+element[1]
        print(sum_value_y)
    print("Final Value",sum_value_x/(len(cluster_1)))
    print("Final Value",sum_value_y/(len(cluster_1)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_1)),sum_value_y/(len(cluster_1))])  
    
    sum_value_x=0
    sum_value_y=0
    
    for i in cluster_2:
        print("Inside third for ==============================================")
        element=feature_dict[i]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
    
    print("Final Value",sum_value_x/(len(cluster_2)))
    print("Final Value",sum_value_y/(len(cluster_2)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_2)),sum_value_y/(len(cluster_2))])  



import math
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
cluster_1=[]
cluster_2=[]
centroid_for_cluster1=feature_dict[1]
centroid_for_cluster2=feature_dict[4]

def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))

temp_count=True

while temp_count :
    print("Inside While============================================================")
    sum_value_x=0
    sum_value_y=0
    prev_cluster_1=cluster_1
    prev_cluster_2=cluster_2
    
    for ele,value in feature_dict.items():
        print("Inside first for ==============================================")
        element=feature_dict[ele]
        print("The key is",ele)
        print("The element is",element)
        check_dist_cent_1=euclidean_dist(element[0],element[1],centroid_for_cluster1[0],centroid_for_cluster1[1])
        print(check_dist_cent_1)
        check_dist_cent_2=euclidean_dist(element[0],element[1],centroid_for_cluster2[0],centroid_for_cluster2[1])
        print(check_dist_cent_2)
        if (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) :
            cluster_1.append(ele)
        elif (check_dist_cent_2<check_dist_cent_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        elif (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) and (ele not in cluster_2):
             cluster_2.append(ele)
        else :
            print("Please check you r program once")
    print(cluster_1)
    print(cluster_2)
    #count=count+1
    
    if cluster_1==prev_cluster_1 and cluster_2==prev_cluster_2:
         temp_count=False
    
    for i in cluster_1:
        print("Inside second for ==============================================")
        print("key",i)
        element=feature_dict[i]
        print("Element",element)
        sum_value_x=sum_value_x+element[0]
        print(sum_value_x)
        sum_value_y=sum_value_y+element[1]
        print(sum_value_y)
    print("Final Value",sum_value_x/(len(cluster_1)))
    print("Final Value",sum_value_y/(len(cluster_1)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_1)),sum_value_y/(len(cluster_1))])  
    
    sum_value_x=0
    sum_value_y=0
    
    for i in cluster_2:
        print("Inside third for ==============================================")
        element=feature_dict[i]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
    
    print("Final Value",sum_value_x/(len(cluster_2)))
    print("Final Value",sum_value_y/(len(cluster_2)))
    centroid_for_cluster1=list([sum_value_x/(len(cluster_2)),sum_value_y/(len(cluster_2))])  


print("The cluster 1",cluster_1)
print("The cluster 2",cluster_2)
import math
import random
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
cluster_1=[]
cluster_2=[]
centroid_for_cluster1=feature_dict[1]
centroid_for_cluster2=feature_dict[4]
list_of_rand_cluster=random.sample(list(feature_dict),2)
print(list_of_rand_cluster)
import math
import random
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
k=int(input("Enter the value for clusters"))
centroid_list=[]
general_cluster=[]
list_of_rand_cluster=random.sample(list(feature_dict),k)
list_of_rand_cluster
for i in list_of_rand_cluster:
    general_cluster.append(list(i))
    element=feature_dict[i]
    centroid_list.append(element)    

list_of_rand_cluster
general_cluster.append(list(i))
for i in list_of_rand_cluster:
    general_cluster.append([i])
    element=feature_dict[i]
    centroid_list.append(element)    

print(list_of_rand_cluster)
print(general_cluster)
print(centroid_list)
print("Random cluster")    
print(list_of_rand_cluster)
print("List of classes")
print(general_cluster)
print("Centroid Value")
print(centroid_list)
def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))

temp_count=True
sum_value_x=0
sum_value_y=0
prev_general_cluster=general_cluster
prev_general_cluster
for ele,value in feature_dict.items():
    #print("Inside first for ==============================================")
    print("The key is",ele)
    print("The element is",value)
    distance=0
    index=0
    count_dist=0
    for i in range(0,len(centroid_list)):
        centroid_value=centroid_list[i]
        print(centroid_value)
        if count_dist==0:
            index=i
            distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
            count_dist=count_dist+1
        else:
            if distance>euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1]):
                distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
                index=i
    general_cluster[index].append(ele)
    if prev_general_cluster==general_cluster:
       temp_count=False
    temp_index=0

import math
import random
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
#cluster_1=[]
#cluster_2=[]
k=int(input("Enter the value for clusters"))
centroid_list=[]
general_cluster=[]
list_of_rand_cluster=random.sample(list(feature_dict),k)
for i in list_of_rand_cluster:
    general_cluster.append([i])
    element=feature_dict[i]
    centroid_list.append(element)

print("Random cluster")    
print(list_of_rand_cluster)
print("List of classes")
print(general_cluster)
print("Centroid Value")
print(centroid_list)
def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))

temp_count=True
sum_value_x=0
sum_value_y=0
prev_general_cluster=general_cluster
for ele,value in feature_dict.items():
    #print("Inside first for ==============================================")
    print("The key is",ele)
    print("The element is",value)
    distance=0
    index=0
    count_dist=0
    for i in range(0,len(centroid_list)):
        print("The centroid value")
        centroid_value=centroid_list[i]
        print(centroid_value)
        if count_dist==0:
            print("Inside if condtion")
            index=i
            print("Index")
            print(index)
            distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
            print("The distance is",distance)
            count_dist=count_dist+1
            print("Count Value")
            print(count_dist)
        else:
            print("Inside else")
            if distance>euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1]):
                print("Inside else if")
                distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
                print("The distance is")
                print(distance)
                print("index")
                index=i
                print(index)


for ele,value in feature_dict.items():
    #print("Inside first for ==============================================")
    print("The key is",ele)
    print("The element is",value)
    distance=0
    index=0
    count_dist=0
    for i in range(0,len(centroid_list)):
        print("The centroid value")
        centroid_value=centroid_list[i]
        print(centroid_value)
        if count_dist==0:
            print("Inside if condtion")
            index=i
            print("Index")
            print(index)
            distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
            print("The distance is",distance)
            count_dist=count_dist+1
            print("Count Value")
            print(count_dist)
        else:
            print("Inside else")
            if distance>euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1]):
                print("Inside else if")
                distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
                print("The distance is")
                print(distance)
                print("index")
                index=i
                print(index)
    
    general_cluster[index].append(ele)

general_cluster
while temp_count :
    print("Inside While============================================================")
    #sum_value_x=0
    #sum_value_y=0
    #prev_cluster_1=cluster_1
    #prev_cluster_2=cluster_2
    sum_value_x=0
    sum_value_y=0
    prev_general_cluster=general_cluster
    for ele,value in feature_dict.items():
        #print("Inside first for ==============================================")
        print("The key is",ele)
        print("The element is",value)
        distance=0
        index=0
        count_dist=0
        for i in range(0,len(centroid_list)):
            print("The centroid value")
            centroid_value=centroid_list[i]
            print(centroid_value)
            if count_dist==0:
                print("Inside if condtion")
                index=i
                print("Index")
                print(index)
                distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
                print("The distance is",distance)
                count_dist=count_dist+1
                print("Count Value")
                print(count_dist)
            else:
                print("Inside else")
                if distance>euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1]):
                    print("Inside else if")
                    distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
                    print("The distance is")
                    print(distance)
                    print("index")
                    index=i
                    print(index)
        if ele not in general_cluster[index]:       
            general_cluster[index].append(ele)

import math
import random
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
#cluster_1=[]
#cluster_2=[]
k=int(input("Enter the value for clusters"))
#centroid_for_cluster1=feature_dict[1]
#centroid_for_cluster2=feature_dict[4]
centroid_list=[]
general_cluster=[]
list_of_rand_cluster=random.sample(list(feature_dict),k)
for i in list_of_rand_cluster:
    general_cluster.append([i])
    element=feature_dict[i]
    centroid_list.append(element)

print("Random cluster")    
print(list_of_rand_cluster)
print("List of classes")
print(general_cluster)
print("Centroid Value")
print(centroid_list)
def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))

temp_count=True
for ele,value in feature_dict.items():
    #print("Inside first for ==============================================")
    print("The key is",ele)
    print("The element is",value)
    distance=0
    index=0
    count_dist=0
    for i in range(0,len(centroid_list)):
        print("The centroid value")
        centroid_value=centroid_list[i]
        print(centroid_value)
        if count_dist==0:
            print("Inside if condtion")
            index=i
            print("Index")
            print(index)
            distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
            print("The distance is",distance)
            count_dist=count_dist+1
            print("Count Value")
            print(count_dist)
        else:
            print("Inside else")
            if distance>euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1]):
                print("Inside else if")
                distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
                print("The distance is")
                print(distance)
                print("index")
                index=i
                print(index)
    if ele not in general_cluster[index]:       
        general_cluster[index].append(ele)


general_cluster
if prev_general_cluster==general_cluster:
   temp_count=False

for i in general_cluster:
    for j in i:
        value_of=feature_dict[i]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
        centroid_list[temp_index]=list([sum_value_x/(len(i)),sum_value_y/(len(i))])
        temp_index=temp_index+1

general_cluster
for i in centroid_list:
    for j in i:
        value_of=feature_dict[i]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
        centroid_list[temp_index]=list([sum_value_x/(len(i)),sum_value_y/(len(i))])
        temp_index=temp_index+1

centroid_list
general_cluster
for i in general_cluster:
    for j in i:
        value_of=feature_dict[int(i)]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
        centroid_list[temp_index]=list([sum_value_x/(len(i)),sum_value_y/(len(i))])
        temp_index=temp_index+1

for i in general_cluster:
    for j in i:
        value_of=feature_dict[j]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
        centroid_list[temp_index]=list([sum_value_x/(len(i)),sum_value_y/(len(i))])
        temp_index=temp_index+1

sum_value_x=0
sum_value_y=0
for i in general_cluster:
    for j in i:
        value_of=feature_dict[j]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
        centroid_list[temp_index]=list([sum_value_x/(len(i)),sum_value_y/(len(i))])
        temp_index=temp_index+1

temp_index=0
for i in general_cluster:
    for j in i:
        value_of=feature_dict[j]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
        centroid_list[temp_index]=list([sum_value_x/(len(i)),sum_value_y/(len(i))])
        temp_index=temp_index+1

temp_index=0
centroid_list_new=[]
for i in general_cluster:
    for j in i:
        value_of=feature_dict[j]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
        centroid_list_new[temp_index]=list([sum_value_x/(len(i)),sum_value_y/(len(i))])
        temp_index=temp_index+1

temp_index=0
centroid_list_new=[]
for i in general_cluster:
    for j in i:
        value_of=feature_dict[j]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
        centroid_list_new[temp_index]=list([sum_value_x/(len(i)),sum_value_y/(len(i))])
        temp_index=temp_index+1

temp_index=0
centroid_list_new=[]
for i in general_cluster:
    for j in i:
        value_of=feature_dict[j]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
        centroid_list_new[temp_index]=list([sum_value_x/(len(i)),sum_value_y/(len(i))])
        temp_index=temp_index+1

for i in general_cluster:
    for j in i:
        value_of=feature_dict[j]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]

for i in general_cluster:
    for j in i:
        value_of=feature_dict[j]
        sum_value_x=sum_value_x+element[0]
        sum_value_y=sum_value_y+element[1]
        centroid_list_new.append([sum_value_x/(len(i)),sum_value_y/(len(i))])
        temp_index=temp_index+1

centroid_list=centroid_list_new
centroid_list
sum_value_x=0
sum_value_y=0
temp_index=0
centroid_list_new=[]
for i in general_cluster:
    print(i)
    for j in i:
        print(j)
        value_of=feature_dict[j]
        print(value_of)
        sum_value_x=sum_value_x+element[0]
        print(sum_value_x)
        sum_value_y=sum_value_y+element[1]
        print(sum_value_x)
        centroid_list_new.append([sum_value_x/(len(i)),sum_value_y/(len(i))])
        temp_index=temp_index+1

sum_value_x=0
sum_value_y=0
temp_index=0
centroid_list_new=[]
for i in general_cluster:
    print(i)
    for j in i:
        print(j)
        value_of=feature_dict[j]
        print(value_of)
        sum_value_x=sum_value_x+value_of[0]
        print(sum_value_x)
        sum_value_y=sum_value_y+value_of[1]
        print(sum_value_x)
        centroid_list_new.append([sum_value_x/(len(i)),sum_value_y/(len(i))])
        temp_index=temp_index+1

general_cluster
import math
import random
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
#cluster_1=[]
#cluster_2=[]
k=int(input("Enter the value for clusters"))
#centroid_for_cluster1=feature_dict[1]
#centroid_for_cluster2=feature_dict[4]
centroid_list=[]
general_cluster=[]
list_of_rand_cluster=random.sample(list(feature_dict),k)
for i in list_of_rand_cluster:
    general_cluster.append([i])
    element=feature_dict[i]
    centroid_list.append(element)

print("Random cluster")    
print(list_of_rand_cluster)
print("List of classes")
print(general_cluster)
print("Centroid Value")
print(centroid_list)
def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))

temp_count=True
sum_value_x=0
sum_value_y=0
prev_general_cluster=general_cluster
for ele,value in feature_dict.items():
    #print("Inside first for ==============================================")
    print("The key is",ele)
    print("The element is",value)
    distance=0
    index=0
    count_dist=0
    for i in range(0,len(centroid_list)):
        print("The centroid value")
        centroid_value=centroid_list[i]
        print(centroid_value)
        if count_dist==0:
            print("Inside if condtion")
            index=i
            print("Index")
            print(index)
            distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
            print("The distance is",distance)
            count_dist=count_dist+1
            print("Count Value")
            print(count_dist)
        else:
            print("Inside else")
            if distance>euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1]):
                print("Inside else if")
                distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
                print("The distance is")
                print(distance)
                print("index")
                index=i
                print(index)
    if ele not in general_cluster[index]:       
        general_cluster[index].append(ele)


if prev_general_cluster==general_cluster:
   temp_count=False

temp_index=0
centroid_list_new=[]
for i in general_cluster:
    print("The value of i")
    print(i)
    for j in i:
        print("The value of j")
        print(j)
        print("The value of")
        value_of=feature_dict[j]
        print(value_of)
        print("sum_value_x")
        sum_value_x=sum_value_x+value_of[0]
        print(sum_value_x)
        sum_value_y=sum_value_y+value_of[1]
        print("sum_value_y")
        print(sum_value_y)
        print("The value of list",[sum_value_x/(len(i)),sum_value_y/(len(i))])
        centroid_list_new.append([sum_value_x/(len(i)),sum_value_y/(len(i))])
        temp_index=temp_index+1

centroid_list=centroid_list_new
centroid_list
centroid_list_new
sum_value_x=0
sum_value_y=0
prev_general_cluster=general_cluster
for ele,value in feature_dict.items():
    #print("Inside first for ==============================================")
    print("The key is",ele)
    print("The element is",value)
    distance=0
    index=0
    count_dist=0
    for i in range(0,len(centroid_list)):
        print("The centroid value")
        centroid_value=centroid_list[i]
        print(centroid_value)
        if count_dist==0:
            print("Inside if condtion")
            index=i
            print("Index")
            print(index)
            distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
            print("The distance is",distance)
            count_dist=count_dist+1
            print("Count Value")
            print(count_dist)
        else:
            print("Inside else")
            if distance>euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1]):
                print("Inside else if")
                distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
                print("The distance is")
                print(distance)
                print("index")
                index=i
                print(index)
    if ele not in general_cluster[index]:       
        general_cluster[index].append(ele)

import math
import random
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
#cluster_1=[]
#cluster_2=[]
k=int(input("Enter the value for clusters"))
#centroid_for_cluster1=feature_dict[1]
#centroid_for_cluster2=feature_dict[4]
centroid_list=[]
general_cluster=[]
list_of_rand_cluster=random.sample(list(feature_dict),k)
for i in list_of_rand_cluster:
    general_cluster.append([i])
    element=feature_dict[i]
    centroid_list.append(element)

print("Random cluster")    
print(list_of_rand_cluster)
print("List of classes")
print(general_cluster)
print("Centroid Value")
print(centroid_list)
def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))

temp_count=True
sum_value_x=0
sum_value_y=0
prev_general_cluster=general_cluster
for ele,value in feature_dict.items():
    #print("Inside first for ==============================================")
    print("The key is",ele)
    print("The element is",value)
    distance=0
    index=0
    count_dist=0
    for i in range(0,len(centroid_list)):
        print("The centroid value")
        centroid_value=centroid_list[i]
        print(centroid_value)
        if count_dist==0:
            print("Inside if condtion")
            index=i
            print("Index")
            print(index)
            distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
            print("The distance is",distance)
            count_dist=count_dist+1
            print("Count Value")
            print(count_dist)
        else:
            print("Inside else")
            if distance>euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1]):
                print("Inside else if")
                distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
                print("The distance is")
                print(distance)
                print("index")
                index=i
                print(index)
    if ele not in general_cluster[index]:       
        general_cluster[index].append(ele)

if prev_general_cluster==general_cluster:
   temp_count=False

temp_index=0
centroid_list_new=[]
temp_index=temp_index+1
temp_index=0
centroid_list_new=[]
for i in general_cluster:
    print("The value of i")
    print(i)
    for j in i:
        print("The value of j")
        print(j)
        print("The value of")
        value_of=feature_dict[j]
        print(value_of)
        print("sum_value_x")
        sum_value_x=sum_value_x+value_of[0]
        print(sum_value_x)
        sum_value_y=sum_value_y+value_of[1]
        print("sum_value_y")
        print(sum_value_y)
        print("The value of list",[sum_value_x/(len(i)),sum_value_y/(len(i))])
    centroid_list_new.append([sum_value_x/(len(i)),sum_value_y/(len(i))])    

centroid_list_new
centroid_list=centroid_list_new
centroid_list
import math
import random
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
#cluster_1=[]
#cluster_2=[]
k=int(input("Enter the value for clusters"))
#centroid_for_cluster1=feature_dict[1]
#centroid_for_cluster2=feature_dict[4]
centroid_list=[]
general_cluster=[]
list_of_rand_cluster=random.sample(list(feature_dict),k)
for i in list_of_rand_cluster:
    general_cluster.append([i])
    element=feature_dict[i]
    centroid_list.append(element)

print("Random cluster")    
print(list_of_rand_cluster)
print("List of classes")
print(general_cluster)
print("Centroid Value")
print(centroid_list)
def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))

temp_count=True

while temp_count :
    print("Inside While============================================================")
    #sum_value_x=0
    #sum_value_y=0
    #prev_cluster_1=cluster_1
    #prev_cluster_2=cluster_2
    sum_value_x=0
    sum_value_y=0
    prev_general_cluster=general_cluster
    for ele,value in feature_dict.items():
        #print("Inside first for ==============================================")
        print("The key is",ele)
        print("The element is",value)
        distance=0
        index=0
        count_dist=0
        for i in range(0,len(centroid_list)):
            print("The centroid value")
            centroid_value=centroid_list[i]
            print(centroid_value)
            if count_dist==0:
                print("Inside if condtion")
                index=i
                print("Index")
                print(index)
                distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
                print("The distance is",distance)
                count_dist=count_dist+1
                print("Count Value")
                print(count_dist)
            else:
                print("Inside else")
                if distance>euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1]):
                    print("Inside else if")
                    distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
                    print("The distance is")
                    print(distance)
                    print("index")
                    index=i
                    print(index)
        if ele not in general_cluster[index]:       
            general_cluster[index].append(ele)
    
    if prev_general_cluster==general_cluster:
       temp_count=False
    
    temp_index=0
    centroid_list_new=[]
    for i in general_cluster:
        print("The value of i")
        print(i)
        for j in i:
            print("The value of j")
            print(j)
            print("The value of")
            value_of=feature_dict[j]
            print(value_of)
            print("sum_value_x")
            sum_value_x=sum_value_x+value_of[0]
            print(sum_value_x)
            sum_value_y=sum_value_y+value_of[1]
            print("sum_value_y")
            print(sum_value_y)
            print("The value of list",[sum_value_x/(len(i)),sum_value_y/(len(i))])
        centroid_list_new.append([sum_value_x/(len(i)),sum_value_y/(len(i))])    
    
    centroid_list=centroid_list_new
        #check_dist_cent_1=euclidean_dist(element[0],element[1],centroid_for_cluster1[0],centroid_for_cluster1[1])
        #print(check_dist_cent_1)
        #check_dist_cent_2=euclidean_dist(element[0],element[1],centroid_for_cluster2[0],centroid_for_cluster2[1])
        #print(check_dist_cent_2)
        #if (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) :
            #cluster_1.append(ele)
        #elif (check_dist_cent_2<check_dist_cent_1) and (ele not in cluster_2):
             #cluster_2.append(ele)
        #elif (check_dist_cent_1<check_dist_cent_2) and (ele not in cluster_1) and (ele not in cluster_2):
             #cluster_2.append(ele)
        #else :
            #print("Please check you r program once")
    #print(cluster_1)
    #print(cluster_2)
    #count=count+1
    
    #if cluster_1==prev_cluster_1 and cluster_2==prev_cluster_2:
         #temp_count=False
    
    #for i in cluster_1:
        #print("Inside second for ==============================================")
        #print("key",i)
        #element=feature_dict[i]
        #print("Element",element)
        #sum_value_x=sum_value_x+element[0]
        #print(sum_value_x)
        #sum_value_y=sum_value_y+element[1]
        #print(sum_value_y)
    #print("Final Value",sum_value_x/(len(cluster_1)))
    #print("Final Value",sum_value_y/(len(cluster_1)))
    #centroid_for_cluster1=list([sum_value_x/(len(cluster_1)),sum_value_y/(len(cluster_1))])  




# =============================================================================
#     for i in cluster_2:
#         print("Inside third for ==============================================")
#         element=feature_dict[i]
#         sum_value_x=sum_value_x+element[0]
#         sum_value_y=sum_value_y+element[1]
#         
#     print("Final Value",sum_value_x/(len(cluster_2)))
#     print("Final Value",sum_value_y/(len(cluster_2)))
#     centroid_for_cluster1=list([sum_value_x/(len(cluster_2)),sum_value_y/(len(cluster_2))])  
#     
# =============================================================================
# =============================================================================
# print("The cluster 1",cluster_1)
# print("The cluster 2",cluster_2)
# 
# =============================================================================
general_cluster
import math
import random
feature_dict={1:[1,2],2:[1.5,2],3:[3.0,4.0],4:[5.0,7.0],5:[3.5,5.0],6:[4.5,5.0],7:[3.5,4.5]}
k=int(input("Enter the value for clusters"))
centroid_list=[]
general_cluster=[]
list_of_rand_cluster=random.sample(list(feature_dict),k)
for i in list_of_rand_cluster:
    general_cluster.append([i])
    element=feature_dict[i]
    centroid_list.append(element)

print("Random cluster")    
print(list_of_rand_cluster)
print("List of classes")
print(general_cluster)
print("Centroid Value")
print(centroid_list)
def euclidean_dist(x_list,y_list,cent_1,cent_2):
     return math.sqrt(math.pow((x_list-cent_1),2)+math.pow((y_list-cent_2),2))

temp_count=True

while temp_count :
    print("Inside While============================================================")
    sum_value_x=0
    sum_value_y=0
    prev_general_cluster=general_cluster
    for ele,value in feature_dict.items():
        #print("Inside first for ==============================================")
        print("The key is",ele)
        print("The element is",value)
        distance=0
        index=0
        count_dist=0
        for i in range(0,len(centroid_list)):
            print("The centroid value")
            centroid_value=centroid_list[i]
            print(centroid_value)
            if count_dist==0:
                print("Inside if condtion")
                index=i
                print("Index")
                print(index)
                distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
                print("The distance is",distance)
                count_dist=count_dist+1
                print("Count Value")
                print(count_dist)
            else:
                print("Inside else")
                if distance>euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1]):
                    print("Inside else if")
                    distance=euclidean_dist(value[0],value[1],centroid_value[0],centroid_value[1])
                    print("The distance is")
                    print(distance)
                    print("index")
                    index=i
                    print(index)
        if ele not in general_cluster[index]:       
            general_cluster[index].append(ele)
    
    if prev_general_cluster==general_cluster:
       temp_count=False
    
    temp_index=0
    centroid_list_new=[]
    for i in general_cluster:
        print("The value of i")
        print(i)
        for j in i:
            print("The value of j")
            print(j)
            print("The value of")
            value_of=feature_dict[j]
            print(value_of)
            print("sum_value_x")
            sum_value_x=sum_value_x+value_of[0]
            print(sum_value_x)
            sum_value_y=sum_value_y+value_of[1]
            print("sum_value_y")
            print(sum_value_y)
            print("The value of list",[sum_value_x/(len(i)),sum_value_y/(len(i))])
        centroid_list_new.append([sum_value_x/(len(i)),sum_value_y/(len(i))])    
    
    centroid_list=centroid_list_new


print(general_cluster)