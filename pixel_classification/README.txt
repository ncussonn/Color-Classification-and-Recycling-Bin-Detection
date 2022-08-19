PIXEL CLASSIFIER - READ ME

generate_rgb_data	- generates training set RGB data for Logistic Regression
LogRegress_Trainer	- trains 3 logistic regression model weights using generated RGB training dataset
pixel_classifier	- main script with contains class PixelClassifier with method classify
test_pixel_classifier	- used to test PixelClassifier method classify on validation testset
w_MLE_iteration 	- test script used to learn how logistic regression works

Training:

Binary logistic regression parameters were trained using the "Trainer" function from
the "LogRegress_Trainer.py" file.

The function accepts a training RGB pixel data set, associated label vector, RGB pixel color, 
gradient descent learning rate, and iteration count as inputs (variable names X,y,color,a,k respectfully).
Trainer returns a value "omega", which is the weighting vector for the associated color.

Trainer function syntax:
Trainer(X,y,color,a,k)

In order to classify RGB pixels, 3 different binary classifier models must be made, one for red, green,
and blue respectfully.

The resulting weights were hard coded into the "classify" function from the "pixel_classifier.py" file.

Inputting a new dataset to be labeled into the classify function will produce a label vector y.

Classify function syntax:
classify(X)