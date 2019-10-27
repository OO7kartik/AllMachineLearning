import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt    # for using fmin_tnc (optimizer)

#Reading the data
data = pd.read_csv("ex2data1.txt",header=None)
X = data.iloc[:,:-1]
y = data.iloc[:,2]


#Visualization
#mask = y == 1
#adm = plt.scatter(X[mask][0].values, X[mask][1].values)
#not_adm = plt.scatter(X[~mask][0].values, X[~mask][1].values)
#plt.xlabel('Exam 1 score')
#plt.ylabel('Exam 2 score')
#plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
#plt.show()

def sigmoid(x):
    return 1/(1+np.exp(-x))

def costFunction(theta,X,y):
    J = (-1/m)*( np.sum(np.multiply(y,np.log(sigmoid(X@theta))) + np.multiply((1-y),np.log(1-sigmoid(X@theta)))))
    return J

def gradient(theta,X,y):
	return ( (1/m)*X.T@(sigmoid(X@theta)-y) ).flatten()

(m, n) = X.shape
X = np.hstack((np.ones((m,1)), X)) #adding Xo
y = y[:, np.newaxis]
theta = np.zeros((n+1,1)) # intializing theta with all zeros
J = costFunction(theta, X, y)
#print(J)


#optimizing
#this gives us thetadirectly
#only the first element of the returned tupe is our optimized theta
temp = opt.fmin_tnc(func = costFunction, 
                    x0 = theta.flatten(),fprime = gradient, 
                    args = (X, y.flatten()))
theta_optimized = temp[0]
#print(theta_optimized)

#now lets check the cost
J = costFunction(theta_optimized[:,np.newaxis], X, y)
print(J)

#ploting the boundary
plot_x = [np.min(X[:,1]-2), np.max(X[:,2]+2)]
plot_y = -1/theta_optimized[2]*(theta_optimized[0] 
          + np.dot(theta_optimized[1],plot_x))  
mask = y.flatten() == 1
adm = plt.scatter(X[mask][:,1], X[mask][:,2])
not_adm = plt.scatter(X[~mask][:,1], X[~mask][:,2])
decision_boun = plt.plot(plot_x, plot_y)
plt.xlabel('Exam 1 score')
plt.ylabel('Exam 2 score')
plt.legend((adm, not_adm), ('Admitted', 'Not admitted'))
plt.show()

#to check the accuracy
def accuracy(X, y, theta, cutoff):
    pred = [sigmoid(np.dot(X, theta)) >= cutoff]
    acc = np.mean(pred == y)
    print(acc * 100)
accuracy(X, y.flatten(), theta_optimized, 0.5)
