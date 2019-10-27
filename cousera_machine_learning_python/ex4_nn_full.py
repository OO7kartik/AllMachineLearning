import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from scipy.io import loadmat # to laod matlab files


data = loadmat("ex4data1") #loading the matlab data
y = data['y']
X = np.c_[np.ones((data['X'].shape[0],1)),data['X']] #adding the extra column of ones

print('X:',X.shape, '(with intercept)') #5000x401
print('y:',y.shape) #5000x1

weights = loadmat("ex4weights") #loading the weights
theta_1,theta_2 = weights['Theta1'],weights['Theta2']

print("dimensions of Theta1: {}".format(theta_1.shape)) #25x401
print("dimensions of Theta2: {}".format(theta_2.shape)) #10x26

params = np.r_[theta_1.ravel(),theta_2.ravel()] #converting all the weights into a single vector since we dont know how to distribute theta1 n theta2
#after passing into the nn funcion we know inputlayer size and hidden_layers size so we can distribute it than
print("size o params ",params.shape) #10285

def sigmoid(z):
	return (1/(1+np.exp(-z)))

#we need this for backpropagation
def sigmoidGradient(z):
	return sigmoid(z)*(1-sigmoid(z))

#input_layer = 400; hidden_layer = 25
def cost(nn_params, input_layer_size, hidden_layer_size, num_labels, features, classes, reg):
    
    # When comparing to Octave code note that Python uses zero-indexed arrays.
    # But because Numpy indexing does not include the right side, the code is the same anyway.
    theta1 = nn_params[0:(hidden_layer_size*(input_layer_size+1))].reshape(hidden_layer_size,(input_layer_size+1))
    theta2 = nn_params[(hidden_layer_size*(input_layer_size+1)):].reshape(num_labels,(hidden_layer_size+1))

    m = features.shape[0]
    y_matrix = pd.get_dummies(classes.ravel()).as_matrix() 
    
    # Cost
    a1 = features # 5000x401
        
    z2 = theta1.dot(a1.T) # 25x401 * 401x5000 = 25x5000 
    a2 = np.c_[np.ones((features.shape[0],1)),sigmoid(z2.T)] # 5000x26 
    
    z3 = theta2.dot(a2.T) # 10x26 * 26x5000 = 10x5000 
    a3 = sigmoid(z3) # 10x5000
    
    J = -1*(1/m)*np.sum((np.log(a3.T)*(y_matrix)+np.log(1-a3).T*(1-y_matrix))) + \
        (reg/(2*m))*(np.sum(np.square(theta1[:,1:])) + np.sum(np.square(theta2[:,1:])))

    # Gradients
    d3 = a3.T - y_matrix # 5000x10
    d2 = theta2[:,1:].T.dot(d3.T)*sigmoidGradient(z2) # 25x10 *10x5000 * 25x5000 = 25x5000
    
    delta1 = d2.dot(a1) # 25x5000 * 5000x401 = 25x401
    delta2 = d3.T.dot(a2) # 10x5000 *5000x26 = 10x26
    
    theta1_ = np.c_[np.ones((theta1.shape[0],1)),theta1[:,1:]]
    theta2_ = np.c_[np.ones((theta2.shape[0],1)),theta2[:,1:]]
    
    theta1_grad = delta1/m + (theta1_*reg)/m
    theta2_grad = delta2/m + (theta2_*reg)/m
    
    return(J, theta1_grad, theta2_grad)

print(cost(params, 400, 25, 10, X, y, 0)[0]) #cost