# -*- coding: utf-8 -*-
"""
Created on Thu Feb  3 14:40:20 2022

@author: Nate
"""

#exec(open("./test_segmenter.py").read())

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import cv2
from bin_detector import BinDetector
my_detector = BinDetector()

img = cv2.imread('./data/training/0048.jpg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
mask_img = my_detector.segment_image(img)

fig, ax = plt.subplots()
h = ax.imshow(mask_img)