print("Enter the age of family of 20 members")
print("Range of children is 0 to 20")
print("Range of young age is 20 to 30")
print("Range of middle aged is 30 to 60")
print("Range of old age is more than 60")
name=[]
age=[]
children=[]
young_age=[]
middle_age=[]
old_age=[]
for i in range(0,20):
    input_name=input("Enter the name of member")
    input_age=int(input("Enter the age of the member"))
    name.append(input_name)
    age.append(input_age)
    if input_age>60:
        print("Old")
        old_age.append(name)
    elif input_age>30 and input_age<=60:
        print("Middle")
        middle_age.append(name)
    elif input_age>20 and input_age<=30:
        print("Young")
        young_age.append(name)
    elif input_age>=0 and input_age<=20:
        print("Children")
        children.append(name)
   
