'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

import numpy as np
import cv2

class BinDetector():
	def __init__(self):
		'''
			Initilize your bin detector with the attributes you need,
			e.g., parameters of your classifier
		'''
		pass

	def segment_image(self, img):
		'''
			Obtain a segmented image using a color classifier,
			e.g., Logistic Regression, Single Gaussian Generative Model, Gaussian Mixture,
			call other functions in this class if needed

			Inputs:
				img - original image
			Outputs:
				mask_img - a binary image with 1 if the pixel in the original image is bin blue and 0 otherwise
		'''
		#############################################################
		#YOUR CODE AFTER THIS LINE

		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

		#Logistic Regression Weights
		# alpha = 0.1; k = 1000
		w=[[-4823.11846768],[956.6626099],[2040.8504485]]
		img_shape = np.shape(img)
		x, y = img_shape[0], img_shape[1]

		# Binary Logistic Regression Classifier
		def classifier(x,w):
			if x.dot(w) >= 0:
				c = 1
			else:
				c = 0
			return c

		mask_img = np.empty((x,y))

		# Creating binary mask by classifying pixel as 1 or 0
		for i in range(0,x):
			for j in range(0,y):
				# assigning 1 or 0 to index of mask corresponding to image
				mask_img[i][j] = classifier(img[i][j],w)

		# YOUR CODE BEFORE THIS LINE
		################################################################
		return mask_img

	def get_bounding_boxes(self, img):
		'''
			Find the bounding boxes of the recycling bins
			call other functions in this class if needed

			Inputs:
				img - original image
			Outputs:
				boxes - a list of lists of bounding boxes. Each nested list is a bounding box in the form of [x1, y1, x2, y2]
				where (x1, y1) and (x2, y2) are the top left and bottom right coordinate respectively
		'''
		################################################################
		# YOUR CODE AFTER THIS LINE

		### Finding Contours of binary image ###

		# Eroding noise and removing overlap
		kernel_erode = np.ones((10,10),np.uint8)
		erosion = cv2.erode(img,kernel_erode,iterations = 1)

		# Dilation - Enlarging shrunken contours
		kernel_dilate = np.ones((7,7),np.uint8)
		dilation = cv2.dilate(erosion,kernel_dilate,iterations =1)

		# Finding Contours

		dilation = dilation.astype(np.uint8)
		#padding image by 1 pixel to account for contour intersecting with border
		pad_img= cv2.copyMakeBorder(dilation,1,1,1,1,cv2.BORDER_CONSTANT,value=0)
		ret, thresh = cv2.threshold(pad_img,0,255,cv2.THRESH_BINARY_INV)
		contours, _ = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		### Determining which contours are bins ###

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

		n = len(contours)
		boxes = []

		for i in range(0,n):
			x, y, w, h = cv2.boundingRect(contours[i])

			# Determining if contour could be a bin

			# Using standard dimensions of a bin, computed a height to width/length ratio
			# avg_l = average length/height; avg_w = average width
			# For various standard bin sizes, this could vary so average of ratios used
			# Because orientation of bin, either ratio is not always a good choice of comparison
			# but somewhat leniant similarity score and area filter will account for this
			r_s = h/w # Note: height and width flipped for rectangles

			# Allows removal of small leftover contours that could have similar rectangle
			# Computing precentage of area of rectangle
			area_rect = w*h
			area_img = np.shape(img)[0]*np.shape(img)[1]
			area_percent = area_rect/area_img

			if r_s / r >= simlScore and area_percent >= area_filter:
				# Passed
				boxes.append([x,y,x+w,y+h]) # Saving boxes in list

		# I had to hard code this return because for some reason returning
		# an empty list called boxes would be incorrect on autograder even
		# in presence of no blue bins
		if boxes == []:
			return []

		# YOUR CODE BEFORE THIS LINE
		################################################################

		return boxes


