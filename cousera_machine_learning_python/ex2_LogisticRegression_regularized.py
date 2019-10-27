import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt    # for using fmin_tnc (optimizer)

#Reading the data
data = pd.read_csv("ex2data2.txt",header=None)
X = data.iloc[:,:-1]
y = data.iloc[:,2]


#Visualization
#mask = y == 1
#passed = plt.scatter(X[mask][0].values, X[mask][1].values)
#failed = plt.scatter(X[~mask][0].values, X[~mask][1].values)
#plt.xlabel('Microchip Test1')
#plt.ylabel('Microchip Test2')
#plt.legend((passed, failed), ('Passed', 'Failed'))
#plt.show()

def mapFeature(X1, X2):
    degree = 6
    out = np.ones(X.shape[0])[:,np.newaxis]
    for i in range(1, degree+1): #1-6
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j),                                     np.power(X2, j))[:,np.newaxis]))
    return out
X = mapFeature(X.iloc[:,0], X.iloc[:,1])

def sigmoid(x):
    return 1/(1+np.exp(-x))

def costFunction(theta_t, X_t, y_t, lambda_t):
    m = len(y_t)
    J = (-1/m) * (y_t.T @ np.log(sigmoid(X_t @ theta_t)) + (1 - y_t.T) @ np.log(1 - sigmoid(X_t @ theta_t)))
    reg = (lambda_t/(2*m)) * (theta_t[1:].T @ theta_t[1:])
    J = J + reg
    return J

def gradient(theta, X, y, lambda_t):
    m = len(y)
    grad = np.zeros([m,1])
    grad = (1/m) * X.T @ (sigmoid(X @ theta) - y)
    grad[1:] = grad[1:] + (lambda_t / m) * theta[1:]
    return grad


(m, n) = X.shape
y = y[:, np.newaxis]
theta = np.zeros((n,1)) # intializing theta with all zeros
lmbda = 1
J = costFunction(theta, X, y,lmbda)
print(J)


#optimizing
#this gives us thetadirectly
#only the first element of the returned tupe is our optimized theta
temp = opt.fmin_tnc(func = costFunction, 
                    x0 = theta.flatten(),fprime = gradient, 
                    args = (X, y.flatten(),lmbda))
theta_optimized = temp[0]
#print(theta_optimized)

#now lets check the cost
J = costFunction(theta_optimized[:,np.newaxis], X, y,lmbda)
print(J)

pred = [sigmoid(np.dot(X, theta_optimized)) >= 0.5]
print(np.mean(pred == y.flatten()) * 100)


#Visualization
u = np.linspace(-1, 1.5, 50)
v = np.linspace(-1, 1.5, 50)
z = np.zeros((len(u), len(v)))
def mapFeatureForPlotting(X1, X2):
    degree = 6
    out = np.ones(1)
    for i in range(1, degree+1):
        for j in range(i+1):
            out = np.hstack((out, np.multiply(np.power(X1, i-j), np.power(X2, j))))
    return out
for i in range(len(u)):
    for j in range(len(v)):
        z[i,j] = np.dot(mapFeatureForPlotting(u[i], v[j]), theta)
mask = y.flatten() == 1
X = data.iloc[:,:-1]
passed = plt.scatter(X[mask][0], X[mask][1])
failed = plt.scatter(X[~mask][0], X[~mask][1])
plt.contour(u,v,z,0)
plt.xlabel('Microchip Test1')
plt.ylabel('Microchip Test2')
plt.legend((passed, failed), ('Passed', 'Failed'))
plt.show()