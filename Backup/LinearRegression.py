import numpy as  np
import matplotlib.pyplot as mp
x=np.array([8,2,11,6,5,4,12,9,6,1])
y=np.array([3,10,3,6,8,12,1,4,9,14])
"""print(x)
print(y)
p1=np.polyfit(x,y,1)
p2=np.polyfit(x,y,2)
print(p1)

mp.plot(x,y,'o')
mp.plot(x,np.polyval(p1,x),'r-')
mp.plot(x,np.polyval(p2,x),'g-')
mp.show()
"""

#Plot through diff method
mean_x=np.mean(x)
mean_y=np.mean(y)
print(mean_x)
print(mean_y)
x_meanx=x-mean_x
y_meany=y-mean_y
print(x_meanx)
print(y_meany)
mul_meanxy=x_meanx*y_meany
print(sum(mul_meanxy))
squarex=x_meanx**2
print(squarex)
print(sum(squarex))
slope=round(sum(mul_meanxy)/sum(squarex),2)
print(slope)
intercept=mean_y-(mean_x*slope)
print(intercept)
new_y=(slope*x)+intercept
print(new_y)
mp.scatter(x,y)
mp.plot(x,new_y,'r')
mp.show()
 
