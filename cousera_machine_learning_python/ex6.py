import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat #to load matlab files
from sklearn.svm import SVC


#functions to visualize
def plotData(X,y):
    pos = (y==1).ravel()
    neg = (y==0).ravel()

    plt.scatter(X[pos,0], X[pos,1], s=60, c='k', marker='+', linewidths=1)
    plt.scatter(X[neg,0], X[neg,1], s=60, c='y', marker='o', linewidths=1)
    plt.show()
#h is the steps we take to plot, pad is the padding from the closest data from the center of the plane,
#sorta margin.
def plot_svc(svc,X,y,h = 0.02,pad = 0.25):
    x_min,x_max = X[:,0].min() - pad, X[:,0].max() + pad
    y_min,y_max  =X[:,1].min()-pad, X[:,1].max()+pad
    xx,yy = np.meshgrid(np.arange(x_min,x_max,h),np.arange(y_min,y_max,h))
    Z = svc.predict(np.c_[xx.ravel(),yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx,yy,Z,cmap=plt.cm.Paired,alpha=0.2)

    plotData(X,y)

    sv = svc.support_vectors_
    plt.scatter(sv[:,0], sv[:,1], c='k', marker='|', s=100, linewidths='1')
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()
    print('Number of support vectors: ', svc.support_.size)



#start :)
#using the linear kernel
data1 = loadmat('ex6data1.mat')

y1 = data1['y']
X1 = data1['X']

print('X1:', X1.shape)
print('y1:', y1.shape)

plotData(X1,y1)

#C basically tells how much miscladdifying to avoid, high c:-smaller margin,low c:-larger margin
clf = SVC(C=1.0,kernel='linear')
clf.fit(X1,y1)
plot_svc(clf,X1,y1)

clf.set_params(C=100)
clf.fit(X1, y1.ravel())
plot_svc(clf, X1, y1)

#using gassian kernel
#sigma is the variance
def gaussianKernel(X1,X2,sigma=2):
	norm = (X1-X2).T.dot(X1-X2)
	return np.exp(-norm/(2*sigma**2))

x1 = np.array([1, 2, 1])
x2 = np.array([0, 4, -1])
sigma = 2

print(gaussianKernel(x1, x2, sigma))

#with the second dataset
#the first was easy a linearly seperable dataset
#not lets try this

data2 = loadmat('ex6data2.mat')

y2 = data2['y']
X2 = data2['X']

print('X2:', X2.shape)
print('y2:', y2.shape)

plotData(X2, y2)

#using sbf radial basis kernel, it uses gamma. google it!
clf2 = SVC(C=50, kernel='rbf', gamma=6)
clf2.fit(X2, y2.ravel())
plot_svc(clf2, X2, y2)


#now with the 3rd dataset..
data3 = loadmat('ex6data3.mat')

y3 = data3['y']
X3 = data3['X']

print('X3:', X3.shape)
print('y3:', y3.shape)

plotData(X3, y3)

clf3 = SVC(C=1.0, kernel='poly', degree=3, gamma=10)
clf3.fit(X3, y3.ravel())
plot_svc(clf3, X3, y3)