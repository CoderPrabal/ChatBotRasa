import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
plt.rc("font", size=14)
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
data = pd.read_csv('C:\\Users\\user\\Desktop\\Adult\\adult_data.csv',header=0)
data.replace(' ?',np.NaN,inplace=True)
print(data)
data = data.dropna()
print(data)
print(data.columns)
sns.countplot(x='Salary', data=data, palette = 'hls')
plt.show()
plt.savefig('count_plot')
pd.crosstab(data['Salary'],data['Age']).plot(kind='bar')
print(data[data['Salary']=='<=50K'])
plt.title('Age Vs Salary')
plt.xlabel('Salary')
plt.ylabel('Age')
plt.savefig('purchase_fre_job')
#trying
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('hist_age')
data["Age"].hist()
"""table=pd.crosstab(data['Salary'],data['Age'])  
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of  WorkClass vs Salary')
plt.xlabel('Education')
plt.ylabel('Proportion of Salary')
plt.savefig('edu_vs_pur_stack')"""
#keep trying
table=pd.crosstab(data['Workclass'],data['Salary'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of  WorkClass vs Salary')
plt.xlabel('WorkClass')
plt.ylabel('Proportion of Salary')
#plt.savefig('edu_vs_pur_stack')
data.loc[data['Salary']==' <=50K', 'Salary']=0
data.loc[data['Salary']==' >50K', 'Salary']=1
list1=sorted(data['Age'].unique())
for i in list1:
    print(data[data['Age']==int(i)]['Salary'].value_counts())

"""plt.title('Histogram of Fnlwgt')
plt.xlabel('Fnlwgt')
plt.ylabel('Frequency')
plt.savefig('hist_age')
data["Fnlwgt"].hist()
"""
table=pd.crosstab(data['Salary'].head(20),data['Fnlwgt'].head(20))
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of  Fnlwgt vs Salary')
plt.xlabel('Fnlwgt')
plt.ylabel('Proportion of Salary')

#plt.savefig('edu_vs_pur_stack')"""
table=pd.crosstab(data['Education'],data['Salary'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of  Education vs Salary')
plt.xlabel('Education')
plt.ylabel('Proportion of Salary')
#next Education_num
table=pd.crosstab(data['Education_num'],data['Salary'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of  Education_num vs Salary')
plt.xlabel('Education')
plt.ylabel('Proportion of Salary')

table=pd.crosstab(data['Maritalstatus'],data['Salary'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of  Maritalstatus vs Salary')
plt.xlabel('Maritalstatus')
plt.ylabel('Proportion of Salary')

table=pd.crosstab(data['Occupation'],data['Salary'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of  Occupation vs Salary')
plt.xlabel('Occupation')
plt.ylabel('Proportion of Salary')

table=pd.crosstab(data['Relationship'],data['Salary'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of  Relationship vs Salary')
plt.xlabel('Relationship')
plt.ylabel('Proportion of Salary')

table=pd.crosstab(data['Race'],data['Salary'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of  Race vs Salary')
plt.xlabel('Race')
plt.ylabel('Proportion of Salary')


table=pd.crosstab(data['Gender'],data['Salary'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of  Gender vs Salary')
plt.xlabel('Gender')
plt.ylabel('Proportion of Salary')


table=pd.crosstab(data['Capitalgain'],data['Salary'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of  Capitalgain vs Salary')
plt.xlabel('Capitalgain')
plt.ylabel('Proportion of Salary')

table=pd.crosstab(data['Capitaloss'],data['Salary'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of  Capitaloss vs Salary')
plt.xlabel('Capitaloss')
plt.ylabel('Proportion of Salary')


table=pd.crosstab(data['Hoursperweek'].head(100),data['Salary'].head(100))
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of  Hoursperweek vs Salary')
plt.xlabel('Hoursperweek')
plt.ylabel('Proportion of Salary')

table=pd.crosstab(data['Native'],data['Salary'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of  Native vs Salary')
plt.xlabel('Native')
plt.ylabel('Proportion of Salary')


#Important property according to data visualization
#Age
#WorkClass
#Education_num
#Occupation
#HoursPerweek

X=data[['Age','Workclass','Education','Education_num','Occupation','Hoursperweek']]
print(X)
#using the new dataframe
cat_vars=['Workclass','Occupation','Education']
for i in cat_vars:
    cat_list = 'var'+'_'+i
    cat_list = pd.get_dummies(X[i],prefix=i)
    x1= X.join(cat_list)
    X=x1

to_keep = [i for i in X.columns if i not in cat_vars]
X_Final = X[to_keep]
Y=data["Salary"]
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import RFE
logistic = LogisticRegression()
rfe = RFE(logistic,17)
rfe.fit(X_Final,Y)
print(rfe.support_)
print(rfe.ranking_)
selected_columns =[]
for i in range(len(rfe.support_)):
    if(rfe.support_[i] == True):
        selected_columns.append(to_keep[i])
 
X_Report = X_Final[selected_columns]  

selected_columns =[]
for i in range(len(rfe.support_)):
    if(rfe.support_[i] == True):
        selected_columns.append(to_keep[i])
 
X_Report = X_Final[selected_columns]       

from scipy import stats
stats.chisqprob = lambda chisq, data: stats.chi2.sf(chisq,data)
import statsmodels.api as sm
logit_model=sm.Logit(Y,X_Report)
result=logit_model.fit()
print(result.summary())

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_Report, Y, test_size=0.99, random_state=0)
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
logreg.fit(X_train, y_train)
    
y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(logreg.score(X_test, y_test)))

from sklearn import model_selection
from sklearn.model_selection import cross_val_score
kfold = model_selection.KFold(n_splits=10, random_state=7)
modelCV = LogisticRegression()
scoring = 'accuracy'
results = model_selection.cross_val_score(modelCV, X_train, y_train, cv=kfold, scoring=scoring)
print("10-fold cross validation average accuracy: %.3f" % (results.mean()))

from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)     







