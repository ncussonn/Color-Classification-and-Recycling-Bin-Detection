'''
ECE276A WI22 PR1: Color Classification and Recycling Bin Detection
'''

#Altered file for use to create pixel data sets
#exec(open("./test_roipoly.py").read())

import os, cv2
from roipoly import RoiPoly
from matplotlib import pyplot as plt
import numpy as np

if __name__ == '__main__':

  # read the first training image
  folder = 'data/training'
  filename = '0029.jpg'
  img = cv2.imread(os.path.join(folder,filename))
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

  # display the image and use roipoly for labeling
  fig, ax = plt.subplots()
  ax.imshow(img)
  my_roi = RoiPoly(fig=fig, ax=ax, color='r')

  # get the image mask
  mask = my_roi.get_mask(img)

  # indices of True values in mask
  mask_index_true = np.where(mask == True)   # returns 2 x X tuple of coordinates

  # Define data array
  n = len(mask_index_true[0])

  # Suppress X for first image
  Y = np.empty([n,3])
  #X = np.load('bluebindata.npy')
  #X = np.load('notbluedata.npy')

  for i in range(0,len(mask_index_true[0])):

      # Coordinates of true pixel
      x = mask_index_true[0][i]
      y = mask_index_true[1][i]

      # Saving value of pixel to use in training
      Y[i] = img[x,y]

  # Append new RGB pixels to training dataset
  X = np.append(X, Y, axis = 0)

  # Save dataset for training later

  #For first image (Suppress after 1st image)
  np.save('bluebindata', Y)
  #np.save('notbluedata.npy',Y)

  # After first
  #np.save('bluebindata',X)
  #np.save('notbluedata',X)

  # Display the labeled region and the image mask
  fig, (ax1, ax2) = plt.subplots(1, 2)
  fig.suptitle('%d pixels selected\n' % img[mask,:].shape[0])

  ax1.imshow(img)
  ax1.add_line(plt.Line2D(my_roi.x + [my_roi.x[0]], my_roi.y + [my_roi.y[0]], color=my_roi.color))
  ax2.imshow(mask)

  plt.show(block=True)

