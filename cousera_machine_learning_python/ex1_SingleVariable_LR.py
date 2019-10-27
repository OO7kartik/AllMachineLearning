#Linear Regression single variable
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#csv file ex1data1 , population,percetage_profit

data = pd.read_csv("ex1data1.txt",header = None) #read the dataset
X = data.iloc[:,0] #only the first column
y = data.iloc[:,1] #only the second column
m = len(y) #number of training examples
#print(data.head())

#Visualization
#plt.scatter(X,y)
#plt.xlabel('Population of City in 10,000s')
#plt.ylabel('Profit in $10,000s')
#plt.show()

# we need to add Xo, for the intercept
X = X[:,np.newaxis] #making this a column vector
y = y[:,np.newaxis]
theta = np.zeros([2,1])
iterations = 1500
alpha = 0.01 #learing rate
ones = np.ones((m,1)) #we need to make a column for (theta)o so Xo should be ones
X = np.hstack((ones,X)) #horizontal stacking the ones, so now a column of ones has been added to our X

def computeCost(X,y,theta):
    err = np.dot(X, theta) - y
    return ( 1/(2*m) ) * np.sum(np.power(err,2))

#J = computeCost(X,y,theta)
#print(J)

def gradientDescent(X,y,theta,alpha,iterations):
    for _ in range(iterations):
      temp = np.dot(X,theta) - y
      temp = np.dot(X.T,temp)
      theta = theta - (alpha/m)*temp
    return theta

theta = gradientDescent(X,y,theta,alpha,iterations)
#print(theta)

#now to check if the cost reduced.
#J = computeCost(X,y,theta)
#print(J)


#lets plot the line
plt.scatter(X[:,1], y)
plt.xlabel('Population of City in 10,000s')
plt.ylabel('Profit in $10,000s')
plt.plot(X[:,1], np.dot(X, theta))
plt.show()