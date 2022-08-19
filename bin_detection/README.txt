BIN DETECTION - READ ME

Scripts

bin_detector - main script housing the class BinDetector with methods segment_image and get_bounding_boxes 
training_script 	- used to easily train weight parameters for logistic regression. Prints calculated weights
			for given alpha and k. Imports Trainer from logRegress_Trainer
test_bin_detector 	- script used to test bin_detector class BinDetector on select images
test_roipoly 		- used to collect pixel training set data and put them in proper form
test_segmenter		- test the segment_image method from BinDetector class in bin_detector
boundingbox_test	- used to create get_bounding_boxes method of BinDetector and to create images for report and debug
generate_binblue_data	- uses data from part 1 and roipoly dataset to create training dataset


Misc Files
bluebindata.npy	- numpy data file where roipoly data was saved when collecting blue bin data
notbluedata.npy	- "" when collecting not blue bin data


Generating Blue Bin Data

Used Roipoly function on a selection of images to create binary image mask. Pixel coordinates of the image file
were saved and then used to get RGB values used in classification. Pixel values were stored in numPy array and
saved in npy file. Prior image RGB pixel numPy array was loaded, then appended, then file overwritten.

Note: Due to the large pixel difference in high resolution photos, data will be biased towards them.

Recycling Bin Blue
Images sourced:
0001
0002
0003
0005
0009
0010
0015
0022
0026
0029
0030
0031
0033
0040
0044
0046
0050

Other Colors:
Images sourced:
0001 (grass, white background, black pants)
0007 (gray bin)
0009 (3 times, red, yellow, green bins)
0012 (black trash bin)
0017 (black shadow)
0019 (yellow lid)
0023 (2 times, 1 for ea purple bin)
0033 (green bin)
0045 (black asphalt, black trash bin)
0051 (yellow bin)
0060 (red brick wall)



