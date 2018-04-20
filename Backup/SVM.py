
#labeling the group
#dividing the group
"""digits=datasets.load_digits()
print(digits.data)
print(digits.target)
print(digits.images[0])
print(len(digits.data))
clf=svm.SVC(gamma=0.001,C=100)
x,y=digits.data[:-1],digits.target[:-1]
clf.fit(x,y)
dig_array=digits.data
print('Predictions:',clf.predict(dig_array[-1].reshape(-1,1)))
plt.imshow(digits.images[-1].reshape(-1,1),cmap=plt.cm.gray_r,interpolation="nearest")
plt.show()"""
# importing scikit learn with make_blobs
"""from sklearn.datasets.samples_generator import make_blobs
 
# creating datasets X containing n_samples
# Y containing two classes
X, Y = make_blobs(n_samples=500, centers=2,
                  random_state=0, cluster_std=0.40)
 
# plotting scatters 
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring');
plt.show()
# creating line space between -1 to 3.5 
xfit = np.linspace(-1, 3.5)
 
# plotting scatter
plt.scatter(X[:, 0], X[:, 1], c=Y, s=50, cmap='spring')
 
# plot a line between the different sets of data
for m, b, d in [(1, 0.65, 0.33), (0.5, 1.6, 0.55), (-0.2, 2.9, 0.2)]:
    yfit = m * xfit + b
    plt.plot(xfit, yfit, '-k')
    plt.fill_between(xfit, yfit - d, yfit + d, edgecolor='none', 
    color='#AAAAAA', alpha=0.4)
 
plt.xlim(-1, 3.5);
plt.show()"""

"""x=datasets.load_digits()
a=np.array(x)
y=a[:-1]
x=x = np.column_stack((x.malignant,x.benign))
print(x.shape)
print(x)
print(y)
from sklearn.svm import SVC 
clf = SVC(kernel='linear')
clf.fit(x, y)
clf.predict(digits.data[-1])"""

import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn import svm

digits = datasets.load_digits()
clf = svm.SVC()
clf = svm.SVC(gamma=0.001, C=100)
X,y = digits.data[:-10],digits.target[:-10]
clf.fit(X,y)
print(clf.predict(digits.data[879:880]))
plt.imshow(digits.images[879], cmap=plt.cm.gray_r, interpolation='nearest')
plt.show()