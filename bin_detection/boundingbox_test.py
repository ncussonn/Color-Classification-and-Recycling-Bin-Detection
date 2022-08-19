# BOUNDING BOX TEST SCRIPT - USED TO DEBUG AND CREATE IMAGES FOR REPORT
"""
Test for checking bounding box script
"""
# For quick runs in terminal
# exec(open("./boundingbox_test.py").read())

import numpy as np
import matplotlib.pyplot as plt; plt.ion()
import cv2
from bin_detector import BinDetector

my_detector = BinDetector()
img = cv2.imread('./data/validation/0070.jpg')
img_RGB = img

#Plot for validation and report
cv2.imshow('Original', img_RGB)
cv2.waitKey(0)

# Creating Binary Image
mask_img = my_detector.segment_image(img)

#Plot Binary Image
cv2.imshow('Input', mask_img)
cv2.waitKey(0)

## Finding Contours of binary image ##

# Eroding noise and removing overlap
kernel_erode = np.ones((15,10),np.uint8)
erosion = cv2.erode(mask_img,kernel_erode,iterations = 1)

#Plot Erosion
cv2.imshow('Erosion', erosion)
cv2.waitKey(0)

# Dilation - Enlarging shrunken contours
kernel_dilate = np.ones((3,5),np.uint8)
dilation = cv2.dilate(erosion,kernel_dilate,iterations =1)
img_dilation = dilation

# Plot Dilation
cv2.imshow('Dilation', img_dilation)
cv2.waitKey(0)

## Finding Contours

# Change dilation datatype to work with opencv methods
dilation = dilation.astype(np.uint8)
#padding image to account for contour interseting with border
pad_img= cv2.copyMakeBorder(dilation,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
ret, thresh = cv2.threshold(pad_img,0,255,cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

aug_img = cv2.drawContours(img, contours, -1, (0,0,255),2)

n = len(contours)

# Computing Standard Transhbin Similarity Score

# Average of various standard bin sizes - from project pdf
avg_h = (36.75+44+46)/3
avg_w = (19+22+23)/3
avg_l = (26+26.5+31.5)/3
avg_rw = (avg_w+avg_l)/2
r = avg_h / avg_rw

# Similarity Score - Bounding Boxes greater than or equal to this value
# will pass and be saved/plotted
simlScore = 0.60
# Filter areas below this percentage
area_filter = 0.015

boxes = []

for i in range(0,n):
	x, y, w, h = cv2.boundingRect(contours[i])
	print('Contour Dims')
	print(x,y,w,h)


	# Determining if contour could be a bin

	# Using standard dimensions of a bin, computed a height to width/length ratio
	# r_l = height / length; r_w = heigh / width
	# For various standard bin sizes, this could vary so average of ratios used:
	# r_l =
	# Because orientation of bin, either ratio is not always a good choice of comparison
	# An average of the two ratios is used for determination
	# r = (r_l + r_w)/2
	r_s = h/w

	# Allows removal of small leftover contours that could have similar rectangle
	# Computing precentage of area of rectangle
	area_rect = w*h
	area_img = np.shape(img)[0]*np.shape(img)[1]
	area_percent = area_rect/area_img

	if r_s / r >= simlScore and area_percent >= 0.015:
		# Passed
		boxes.append([x,y,x+w,y+h]) # Saving boxes in list
		aug_img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),2)

print(boxes)

cv2.imshow('Final Image', aug_img)
cv2.waitKey(0)