# -*- coding: utf-8 -*-
"""
Created on Fri Jan 28 15:11:42 2022

@author: Nate
"""

# Training w_MLE

import numpy as np

def s(z):
    sigmoid = np.divide(1,(1+np.exp(-z)));
    
    return sigmoid

# Learning Rate
a = 0.1;

X = np.array([[-3,9,1],[-2,4,1],[-1,1,1],[0,0,1],[1,1,1],[3,9,1]])
y = np.transpose(np.array([[1,1,-1,-1,-1,1]]));
y_star = np.zeros((len(X),1));
omega = np.zeros((len(X[0]),1), dtype=float);

yX = np.multiply(y,X);

# Iteration Count
k = 25;

# Iterate k amount of times:
for i in range(0,k):
    
    sigmoid = s(yX @ omega);
    #sigmoid = np.divide(1,(1+np.exp(-yX @ omega)));
    Sum = np.array([sum(np.multiply(yX,(1-sigmoid)))]);
    omega = omega + a*np.transpose(Sum);
              
print("Final Omega")
print(omega)

for j in range(0,len(y)):
    
    if X[j].dot(omega) >= 0:
        y_star[j] = 1
    else:
        y_star[j] = -1
        
print("Predicted Values")
print(y_star)
print("Finished")   