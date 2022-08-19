# BINARY LOGISTIC REGRESSION TRAINING FUNCTION
"""
Defines the function Trainer for use in training script
"""
# For quick training
import numpy as np

def Trainer(X,y,a,k):
    
    # y must be changed to an immuatble object to avoid altering for other trainings
    y = tuple(y)
    y = np.asarray(y)
    y = np.reshape(y,(len(y),1))
        
    def s(z):
        sigmoid = np.divide(1,(1+np.exp(-z)))
        return sigmoid
    
    omega = np.zeros((len(X[0]),1), dtype=float)
    
    yX = np.multiply(y,X)
    
    # Gradient Descent - k amount of times:
    for j in range(0,k):
        
        sigmoid = s(yX @ omega)
        Sum = np.array([sum(np.multiply(yX,(1-sigmoid)))])
        omega = omega + a*np.transpose(Sum)
                  
    return omega