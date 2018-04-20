#camera_dataset.csv
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)
camera_dataset = pd.read_csv("C:\\Users\\user\\.spyder-py3\\camera_dataset.csv")
print(camera_dataset)
#print the properties of the dataset
print("The properties of the dataset are ")
print(camera_dataset.columns)
print("The datatypes for every columns are")
print(camera_dataset.dtypes)
#data of first 25 entries
print(camera_dataset.head(25))
#summary of the dataset
print(camera_dataset.describe())
#summary stats for price
print(camera_dataset['Price'].describe())
price_greater_than_1000=camera_dataset[camera_dataset['Price']>1000]
print(price_greater_than_1000)
release_date=price_greater_than_1000['Release date']
price_value=price_greater_than_1000['Price']
#graph plot between 
plt.plot(release_date.head(100),price_value.head(100))
plt.xlabel("Release Date")
plt.ylabel("Price Value")
plt.legend("Release Date vs Price Value")
plt.show()
#finding the corr between all the data for exploratory dataanalysis
print(camera_dataset.corr())
#Plotting the graph between price and model
plt.plot(camera_dataset['Model'],camera_dataset['Price'],marker='o')
plt.xlabel("Model")
plt.ylabel("Price")
plt.show()
#first increase happens
plt.plot(camera_dataset['Model'].head(5),camera_dataset['Price'].head(5),marker='o')
plt.xlabel("Model")
plt.ylabel("Price")
plt.show()

plt.plot(camera_dataset['Release date'],camera_dataset['Price'],marker='o')
plt.xlabel("Model")
plt.ylabel("Price")
plt.show()
#first increase happens
plt.plot(camera_dataset['Release date'].head(5),camera_dataset['Price'].head(5),marker='o')
plt.xlabel("Model")
plt.ylabel("Price")
plt.show()

sns.countplot(x=camera_dataset['Price'],data=camera_dataset, palette='hls')
plt.show()
plt.savefig('count_plot')

sns.countplot(x=camera_dataset['Price'].head(60),data=camera_dataset, palette='hls')
plt.show()
plt.savefig('count_plot1')

table=pd.crosstab(camera_dataset['Release date'].head(100),camera_dataset['Price'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Release Date Vs Price')
plt.xlabel('Release Date')
plt.ylabel('Price')
#plt.savefig('mariral_vs_pur_stack')
table=pd.crosstab(camera_dataset['Max resolution'].head(100),camera_dataset['Price'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Max resolution Vs Price')
plt.xlabel('Max resolution')
plt.ylabel('Price')
#
pd.crosstab(camera_dataset['Max resolution'].head(80),camera_dataset['Price']).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Max resolution')
plt.ylabel('Price')
#plt.savefig('purchase_fre_job')
table=pd.crosstab(camera_dataset['Low resolution'].head(100),camera_dataset['Price'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Low resolution Vs Price')
plt.xlabel('Low resolution')
plt.ylabel('Price')
#
pd.crosstab(camera_dataset['Low resolution'].head(80),camera_dataset['Price']).plot(kind='bar')
plt.title('Low resolution for Job Title')
plt.xlabel('Low resolution')
plt.ylabel('Price')
#
table=pd.crosstab(camera_dataset['Low resolution'].head(100),camera_dataset['Effective pixels'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Low resolution Vs Effective pixels')
plt.xlabel('Low resolution')
plt.ylabel('Effective pixels')
#
pd.crosstab(camera_dataset['Low resolution'].head(80),camera_dataset['Effective pixels']).plot(kind='bar')
plt.title('Low resolution for Job Title')
plt.xlabel('Low resolution')
plt.ylabel('Effective pixels')
#
table=pd.crosstab(camera_dataset['Max resolution'].head(100),camera_dataset['Effective pixels'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Max resolution Vs Effective pixels')
plt.xlabel('Max resolution')
plt.ylabel('Effective pixels')
#
pd.crosstab(camera_dataset['Max resolution'].head(80),camera_dataset['Effective pixels']).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Max resolution')
plt.ylabel('Effective pixels')
#
table=pd.crosstab(camera_dataset['Max resolution'].head(100),camera_dataset['Effective pixels'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Max resolution Vs Effective pixels')
plt.xlabel('Max resolution')
plt.ylabel('Effective pixels')
#
pd.crosstab(camera_dataset['Max resolution'].head(80),camera_dataset['Effective pixels']).plot(kind='bar')
plt.title('Max resolution vs Effective pixels')
plt.xlabel('Max resolution')
plt.ylabel('Effective pixels')
#
table=pd.crosstab(camera_dataset['Price'].head(100),camera_dataset['Dimensions'])
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Dimensions Vs Price')
plt.xlabel('Price')
plt.ylabel('Dimensions')
#
pd.crosstab(camera_dataset['Price'].head(80),camera_dataset['Dimensions']).plot(kind='bar')
plt.title('Price vs Dimensions')
plt.xlabel('Dimensions')
plt.ylabel('Price')
#Weight (inc. batteries)
table=pd.crosstab(camera_dataset['Weight (inc. batteries)'],camera_dataset['Price'].head(100))
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Weight (inc. batteries) Vs Price')
plt.xlabel('Weight (inc. batteries)')
plt.ylabel('Price')
#
pd.crosstab(camera_dataset['Weight (inc. batteries)'].head(50),camera_dataset['Price']).plot(kind='bar')
plt.title('Weight (inc. batteries) vs Price')
plt.xlabel('Weight (inc. batteries)')
plt.ylabel('Price')