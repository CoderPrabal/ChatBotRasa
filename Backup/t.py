import pandas as pd
data = pd.read_csv('banking.csv', header=0)
data = data.dropna()
other=['marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
cat_vars=['job']
print("Printing the list")
print(cat_vars)
for var in cat_vars:
    print("Printing the var")
    print(var)
    cat_list='var'+'_'+var
    print("Printing the cat_list")
    print(cat_list)
    cat_list = pd.get_dummies(data[var], prefix=var)
    print("cat list after getdummies")
    print(cat_list)
    data1=data.join(cat_list)
    print(data1)
    data=data1