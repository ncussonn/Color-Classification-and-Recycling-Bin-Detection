#TRAINING SCRIPT
"""
Used to generate weights for classification model
"""
#exec(open("./training_script.py").read())

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import os, cv2
from LogRegress_Trainer import Trainer

# X and y values generated
exec(open("./generate_binblue_data.py").read())

# Learning Rate
a = 0.1

# Gradient Descent Iterations
k = 1000

w = Trainer(X,y,a,k)

print('Weights:')
print(w)


# Generated weights:

''' Generated weights

alpha: 0.1
iterations: 1000

Test Set 1:
Weights:
[[-4823.11846768]
 [  956.6626099 ]
 [ 2040.8504485 ]]

'''