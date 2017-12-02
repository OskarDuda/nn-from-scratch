#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 18:41:53 2017

@author: oskar
"""
import matplotlib.pyplot as plt
import numpy as np
import sklearn.datasets
import seaborn as sns

#Gradient descent parameters
epsilon = 0.01 #learning rate for gradient descent
reg_lambda = 0.01 #regularization strength

class activationFunctionBase():
    def function():
        pass
    def derivative():
        pass
    
class relu(activationFunctionBase):
    def function(z):
        #Apply relu function
        return np.maximum(z,0)
    def derivative(z):
        #Relu derivative
        return np.sign(z)
        
class tanh(activationFunctionBase):
    def function(z,c=1):
        return np.tanh(c*z)
    def derivative(z,c=1):
        return 1 - np.square(np.tanh(c*z))
        
    
class sigmoid(activationFunctionBase):
    def function(z,c=1):
        #Apply sigmoid function
        return 1/(1+np.exp(-z))
    def derivative(z,c=1):
        #Derivative of Sigmoid Function
        return np.exp(-z*c)/(np.power(1+np.exp(-z*c),2))
    
class Neural_Network(object):
    def __init__(self, 
                 inputLayerSize,
                 hiddenLayerSize,
                 outputLayerSize,
                 activation=sigmoid):
        #Define hyperparameters
        self.inputLayerSize = inputLayerSize
        self.hiddenLayerSize = hiddenLayerSize
        self.outputLayerSize = outputLayerSize
        self.activation = activation
        self.scalar = 3
        
        #Weights (Parameters)
        self.W1 = np.random.randn(self.inputLayerSize, 
                                  self.hiddenLayerSize)
        self.W2 =np.random.randn(self.hiddenLayerSize,
                                  self.outputLayerSize)
        
    def forward(self, X):
        #Propagate inputs throught network
        self.z2 = X*self.W1
        self.a2 = self.activation.function(self.z2)
        self.z3 = np.dot(self.a2, self.W2)
        yHat = self.activation.function(self.z3,c=0.5)
        return yHat
    
    def costFunction(self, X, y):
        #Compute cost for given X,y, use weights already stored in class.
        self.yHat = self.forward(X)
        J = 0.5*np.sum(np.power((y-self.yHat),2))
        return J
    
    def costFunctionPrime(self, X, y):
        self.yHat = self.forward(X)
        
        delta3 = np.multiply(-(y-self.yHat), self.activation.derivative(self.z3))
        dJdW2 = -np.dot(self.a2.T, delta3)
        
        delta2 = np.multiply(np.dot(delta3, self.W2.T),self.activation.derivative(self.z2))
        dJdW1 = -np.dot(X.T, delta2)
        
        return dJdW1, dJdW2
    
    def train(self, X, y, n=1, print_est = False, print_cf = False):
        for i in range(n):
           dJdW1, dJdW2 = self.costFunctionPrime(X,y)
           self.W1 = self.W1 + self.scalar*dJdW1
           self.W2 = self.W2 + self.scalar*dJdW2
           if print_est:
               print("Value to estimate:\n {}\n".format(y)+
                     "Estimated value:\n {} \n\n".format(self.forward(X)))
           if print_cf:
               print("Cost function: {}".format(self.costFunction(X,y)))
  
#X = np.matrix([[3,5],[5,1],[10,2],[6,3],[7,4],[7,4]])
#y = np.matrix([[0.75],[0.82],[0.93],[0.85],[0.89],[0.89]])

cmap = sns.dark_palette("green",as_cmap = True)

##Some sample data
##np.random.seed(0)
X, y = sklearn.datasets.make_moons(30, noise=0.12)
X, y = np.matrix(X), np.matrix(y).T.astype(float)
f1 = plt.figure()
f1
plt.scatter(np.array(X[:,0]), np.array(X[:,1]), s=40, c=np.array(y), cmap=cmap)

num_examples = len(X) #training set size
nn_input_dim = 2 #input layer dimensionality
nn_output_dim = 2 #output layer dimensionality

NN = Neural_Network(2,3,1,sigmoid)

min_cost = np.inf
best_params = [NN.W1,NN.W2]
best_iteration = 0
v = []
n = 10000
for i in range(n):
    NN.train(X, y, n=1)
    if NN.costFunction(X,y) < min_cost:
        best_params = [NN.W1,NN.W2]
        best_iteration = i
    v.append(NN.costFunction(X,y))
    
#plt.plot(v)
f2 = plt.figure()
plt.scatter(np.array(X[:,0]), np.array(X[:,1]), s=40, c=np.array(NN.forward(X)), cmap=cmap)