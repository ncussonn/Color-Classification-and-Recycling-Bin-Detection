'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

#exec(open("./test_pixel_classifier.py").read())

import numpy as np

class PixelClassifier():
  def __init__(self):
    '''
	    Initilize your classifier with any parameters and attributes you need
    '''
    pass
	
  def classify(self,X):
    '''
	    Classify a set of pixels into red, green, or blue
	    
	    Inputs:
	      X: n x 3 matrix of RGB values
	    Outputs:
	      y: n x 1 vector of with {1,2,3} values corresponding to {red, green, blue}, respectively
    '''
    ################################################################
    # YOUR CODE AFTER THIS LINE
    
    # Binary Logistic Regression Classifier
   
    # Trained Weights from LogRegress_Trainer.py function Trainer
    w_r = np.array([[ 67.41798294], [-37.7773093],  [-36.43708832]])
    w_g = np.array([[-46.04662991], [ 51.92747637], [-41.01048906]])
    w_b = np.array([[-44.36575161], [-39.83446057], [ 49.82698921]])
    
    # Defining Some Variables
    p_r = p_g = p_b = 0
    
    # Discriminative Model
    
    # Defining Variables
    y   = np.zeros((len(X),1));
    y_r = np.zeros((len(X),1));
    y_g = np.zeros((len(X),1));
    y_b = np.zeros((len(X),1));
             
    for j in range(0,len(X)):

        #Assigning Predicted Values
        
        #red
        if X[j].dot(w_r) >= 0:
            y_r[j] = 1
        else: 
            y_r[j] = -1
        #green
        if X[j].dot(w_g) >= 0:
            y_g[j] = 1
        else: 
            y_g[j] = -1
        #blue
        if X[j].dot(w_b) >= 0:
            y_b[j] = 1
        else: 
            y_b[j] = -1
        
    # Determining which color is most probable when predictions conflict
    
    # Sigmoid Function
    def s(z):
        sigmoid = np.divide(1,(1+np.exp(-z)));
        return sigmoid
     
    for i in range(0,len(X)):
       
        if y_r[j] == 1:
            p_r = s(y_r[i]*X[i] @ w_r)
        else:
            p_r = 0
        
        if y_g[j] == 1:
            p_g = s(y_g[i]*X[i] @ w_g)
        else:
            p_g = 0
            
        if y_b[j] == 1:
            p_b = s(y_b[i]*X[i] @ w_b)
        else:
            p_b = 0
        
        # In the case where no color can be discriminated (all return -1), we look
        # at the least likely prediction to be the color. (i.e. lowest probability that assigning
        # -1 was correct)
        
        if p_r == p_g == p_b == 0:
            # Reassign Values
            p_r = s(y_r[i]*X[i] @ w_r)
            p_g = s(y_g[i]*X[i] @ w_g)
            p_b = s(y_b[i]*X[i] @ w_b)
            p = [p_r,p_g,p_b]
            
            # We want the lowest probability
            pmax = np.argmin(p)
        
        # The Typical Case
        else:
            p = [p_r,p_g,p_b]
            # Determining Highest Probability for i-th pixel
            pmax = np.argmax(p)
        
        # Note: in the very rare case of equal probabilities, argmax will pick lower indexed color that is tied
        # For instance, if r,g,b all equal likelihood, red will be assigned and if g and b equal, green assigned
        if pmax == 0:
            # Pixel is red
            y[i] = 1
        elif pmax == 1:
            #Pixel is green
            y[i] = 2
        elif pmax == 2:
            #Pixel is blue
            y[i] = 3
           
    print('Validation Test:')
    print(y)
    
    # YOUR CODE BEFORE THIS LINE
    ################################################################
    return y

