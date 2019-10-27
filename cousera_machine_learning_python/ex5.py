#bug line 59 trainLR not working.... :(


#linear regression, we want to find the proper polynomial degree upto which we can get a good fitting
#also to find how much data to train on to optimize the models accuracy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat #for loading matlab files
from scipy.optimize import minimize #for directly calculating the theta values

from sklearn.linear_model import LinearRegression,Ridge #for directly using gradient descent
from sklearn.preprocessing import PolynomialFeatures

data = loadmat('ex5data1.mat')

y_train = data['y']
X_train = np.c_[np.ones_like(data['X']), data['X']]

yval = data['yval']
Xval = np.c_[np.ones_like(data['Xval']), data['Xval']]

print('X_train:', X_train.shape)
print('y_train:', y_train.shape)
print('Xval:', Xval.shape)
print('yval:', yval.shape)

plt.scatter(X_train[:,1], y_train, s=50, c='r', marker='x', linewidths=1)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.ylim(ymin=0);


def costfn(theta,X,y,reg):
    m = y.size

    h = X.dot(theta)

    J = (1/(2*m))*(np.sum(np.square(h-y))) + (reg/2*m)*(np.sum(np.square(theta[1:])))

    return J

def gradientfn(theta,X,y,reg):
 	m = y.size

 	h = X.dot(theta)

 	grad = (1/m)*(X.T.dot(h-y)) + (reg/m)*np.r_[[[0]],theta[1:].reshape(-1,1)]

 	return grad.flatten()

initial_theta = np.ones((X_train.shape[1],1)) #not sure why ones... (if kept zeros minimize does not converge)
cost = costfn(initial_theta,X_train,y_train,0)
gradient = gradientfn(initial_theta,X_train,y_train,0)
print(cost)
print(gradient)

def trainLR(X, y, reg):
    #initial_theta = np.ones((X.shape[1],1))
    initial_theta = np.array([[15],[15]])

    res = minimize(costfn, initial_theta, args=(X,y,reg), method=None, jac=gradientfn,
                   options={'maxiter':5000})
    
    return(res)

fit = trainLR(X_train,y_train,0)
print(fit)

regr = LinearRegression(fit_intercept=False) #here fit_intercept = False means that it will have an intercept value of zero
regr.fit(X_train,y_train.ravel())
print(regr.coef_)
print(costnf(regr.coef_,X_train,y_train,0))

plt.plot(np.linspace(-50,40), (fit.x[0]+ (fit.x[1]*np.linspace(-50,40))), label='Scipy optimize')
#plt.plot(np.linspace(-50,40), (regr.coef_[0]+ (regr.coef_[1]*np.linspace(-50,40))), label='Scikit-learn')
plt.scatter(X_train[:,1], y_train, s=50, c='r', marker='x', linewidths=1)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.ylim(ymin=-5)
plt.xlim(xmin=-50)
plt.legend(loc=4);

#to determine how much training size is optimal
def learningCurve(X,y,Xval,yval,reg):
	m = y.size

	error_train = np.zeros((m,1))
	error_val = np.zeros((m,1))

	for i in range(m):
		res = trainLR(X[:i+1],y[:i+1],reg)
		error_train[i] = costfn(res.x,X[:i+1],y[:i+1],reg)
		error_val[i] = costfn(res.x,Xval,yval,reg)
		
		return error_train,error_val


plt.plot(np.arange(1,13), t_error, label='Training error')
plt.plot(np.arange(1,13), v_error, label='Validation error')
plt.title('Learning curve for linear regression')
plt.xlabel('Number of training examples')
plt.ylabel('Error')
plt.legend();



#to decide which degree is the best for our model on this training set
ploy = PloynomialFeatures(degree=8)#max degree
X_train_poly = poly.fit_transform(X_trian[:,1].reshape(-1,1)) #apply the transformation to make X_train_poly hold columns of different degrees

regr2 = LinearRegression() #our model to check for different degrees
regr2.fit(X_train_poly,y_train)

regr3 = Ridge(apha = 20) #another way to use a kind of linear regression
regr.fit(X_train_poly,y_train)

# plot range for x
plot_x = np.linspace(-60,45)
# using coefficients to calculate y
plot_y = regr2.intercept_+ np.sum(regr2.coef_*poly.fit_transform(plot_x.reshape(-1,1)), axis=1)
plot_y2 = regr3.intercept_ + np.sum(regr3.coef_*poly.fit_transform(plot_x.reshape(-1,1)), axis=1)

plt.plot(plot_x, plot_y, label='Scikit-learn LinearRegression')
plt.plot(plot_x, plot_y2, label='Scikit-learn Ridge (alpha={})'.format(regr3.alpha))
plt.scatter(X_train[:,1], y_train, s=50, c='r', marker='x', linewidths=1)
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.title('Polynomial regression degree 8')
plt.legend(loc=4);