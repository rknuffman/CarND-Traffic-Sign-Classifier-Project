# **Traffic Sign Recognition**

## Writeup

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

[5signs]: ./figs/5signs.png "5 Random Traffic Signs"
[aug]: ./figs/augmentations.png "Image Augmentations"
[batch0]: ./figs/batch0.png "Sample Image Class"
[batch1]: ./figs/batch1.png "Sample Image Class"
[batch2]: ./figs/batch2.png "Sample Image Class"
[batch3]: ./figs/batch3.png "Sample Image Class"
[batch4]: ./figs/batch4.png "Sample Image Class"
[batch5]: ./figs/batch5.png "Sample Image Class"
[batch6]: ./figs/batch6.png "Sample Image Class"
[batch7]: ./figs/batch7.png "Sample Image Class"
[batch8]: ./figs/batch8.png "Sample Image Class"
[batch9]: ./figs/batch9.png "Sample Image Class"
[hist_trn]: ./figs/hist_train.png "Training Histogram"
[hist_val]: ./figs/hist_valid.png "Validation Histogram"
[hist_tst]: ./figs/hist_test.png "Testing Histogram"
[hist_aug]: ./figs/hist_aug.png "Augmented Histogram"
[rgb2gray]: ./figs/rgb2gray.png "Grayscale conversion"
[softmax0]: ./figs/softmax0.png "Softmax Output"
[softmax1]: ./figs/softmax1.png "Softmax Output"
[softmax2]: ./figs/softmax2.png "Softmax Output"
[softmax3]: ./figs/softmax3.png "Softmax Output"
[softmax4]: ./figs/softmax4.png "Softmax Output"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

You're reading it! and here is a link to my [project code](https://github.com/rknuffman/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

#### 1. Basic Summary

I used the NumPy library to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34,799
* The size of the validation set is 4,410
* The size of test set is 12,630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

Following are sample image selections from the training data set:

![alt text][batch0]
![alt text][batch1]
![alt text][batch2]
![alt text][batch3]
![alt text][batch4]
![alt text][batch5]

#### 2. Exploratory Visualization

Here is an exploratory visualization of the training, validation, and testing data sets. It is a histogram showing the extent of class imbalance present in the data.

![alt text][hist_trn]
![alt text][hist_val]
![alt text][hist_tst]



### Design and Test a Model Architecture

#### 1. Preprocessing the image data

As a preprocessing step, I converted the images to grayscale and normalized pixels values to a range of (0, 1).  

Here is an example of a traffic sign image before and after converting to grayscale and normalizing.

![alt text][rgb2gray]

Additionally, I decided to generate additional data to compensate for the considerable class imbalance.

For each class with less than 1,000 training images, I generated additional training images to give a minimum of 1,000 per class.  Generated images were created using random combinations of transformations to sampled images within each class.

  * Rotations (+/- 10 degrees)
  * Gaussian Noise
  * Vertical / Horizontal translations (2 px)

Here is an example of an original image and several augmented images:

![alt text][aug]

The difference between the original training data set and the augmented training data set is evidenced in the following side-by-side:

![alt text][hist_trn]
![alt text][hist_aug]

#### 2. Final Model Architecture

My final model consisted of the standard LeNet architecture with the addition of dropout following the first two fully connected layers:

| Layer         		|     Description	        		 |
|:-----------------:|:----------------------------:|
| Input         		| 32x32x1 grayscale image      |
| Convolution 5x5  	| 1x1 stride, outputs 28x28x6  |
| RELU					    |												       |
| Max pooling	2x2	  | 2x2 stride,  outputs 14x14x6 |
| Convolution 5x5   | 1x1 stride, outputs 10x10x16 |
| RELU					    |												       |
| Max pooling	2x2	  | 2x2 stride,  outputs 5x5x16  |
| Flatten           | outputs 400                  |
| Fully Connected   | outputs 120                  |
| RELU					    |												       |
| Dropout           | keep probability = 0.7       |
| Fully connected		| outputs 100	                 |
| RELU					    |												       |
| Dropout           | keep probability = 0.7       |
| Fully Connected   | outputs 43                   |
| Softmax				    | outputs 43	                 |



#### 3. Training - Hyperparameters

To train the model, I used an Adam optimizer with fairly default settings:
  * a learning rate of 0.001
  * a batch size of 64
  * 20 epochs.

#### 4. Training - Approaches

My final model results were:
* training set accuracy of 99.7%
* validation set accuracy of 94.4%
* test set accuracy of 93.4%

My first attempts were using standard hyperparameters, no dropout, and no image augmentation.  In this way I could consistently achieve training accuracy in the mid 90's, and validation accuracy between 89-90%.  This provided a nice baseline performance to start strategizing where improvements could be made.  

Since training accuracy could still improve, I worried that the class imbalance was having a negative impact.  After augmenting my training data with the addition of some randomly transformed images, I retrained my network and observed training accuracy increase to +99%, while my validation accuracy increased only slightly to 90-91%.  

At this point, I was comfortable that my network had enough predictive power to tackle the problem, though appeared to be overfitting the training set.  My next addition was to add some regularization to the training process.  My first attempt was L2 regularization, though, I observed little to no impact on training or validation accuracy.  

Next, I incorporated dropout, and saw validation accuracy jump to the final observed levels of 94-95%.

### Test  on New Images

#### 1. Five German traffic signs.

Here are five German traffic signs that I found on the web:

![alt text][5signs]

* The first image doesn't appear to be too challenging.
* The second image could be challenging to classify since the background strong blends in with the border color of the sign.
* The third and fifth images are examples of several triangular signs with red borders, and various images centered on the sign.  Given the low resolution, it could be challenging to accurately determine the center figure.
* The fourth image doesn't appear too challenging, though the resolution of the image causes finer details like arrows to blur significantly.


#### 2. Model's predictions

Here are the results of the prediction:

| Image			        | Prediction	       					|
|:-----------------:|:---------------------------:|
| No Entry      		| No Entry   									|
| Priority     			| Priority 										|
| Right-of-way		  | Right-of-way								|
| Roundabout     		| Roundabout					 				|
| Ice/snow			    | Right-of-way   							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of 93.4%.

#### 3. Certainty

For the first image, the model is quite sure that this is a 'No Entry' sign (probability of 1.0), and the model is correct. The top five soft max probabilities were

| Probability |     Prediction	    				|
|:-----------:|:---------------------------:|
| 1.0       	| No Entry   									|
| .00     	  | No Passing									|
| .00					| End of No Passing						|
| .00	      	| Priority  					 				|
| .00				  | Dangerous curve right				|


For the second image, the model is quite sure that this is a 'Priority' sign (probability of 1.0), and the model is correct. The top five soft max probabilities were

| Probability |     Prediction	    				|
|:-----------:|:---------------------------:|
| 1.0       	| Priority   									|
| .00     	  | Ahead only									|
| .00					| Roundabout       						|
| .00	      	| Keep right 					 				|
| .00				  | End of No Passing   				|

For the third image, the model is quite sure that this is a 'Right-of-way' sign (probability of 1.0), and the model is correct. The top five soft max probabilities were

| Probability |     Prediction	    				|
|:-----------:|:---------------------------:|
| 1.0       	| Right-of-way								|
| .00     	  | Pedestrians									|
| .00					| Ice/snow        						|
| .00	      	| Double curve				 				|
| .00				  | Priority            				|

For the fourth image, the model is quite sure that this is a 'Roundabout' sign (probability of 1.0), and the model is correct. The top five soft max probabilities were

| Probability |     Prediction	    				|
|:-----------:|:---------------------------:|
| 1.0       	| Roundabout 									|
| .00     	  | 100 km/h  									|
| .00					| No passing 3.5 tons					|
| .00	      	| Priority  					 				|
| .00				  | Go Straight or Left 				|

For the fifth image, the model is quite sure that this is a 'Right-of-way' sign (probability of 1.0), but the model is incorrect. The top five soft max probabilities were

| Probability |     Prediction	    				|
|:-----------:|:---------------------------:|
| .89       	| Right-of-way								|
| .11     	  | Ice/snow   									|
| .00					| Children crossing 					|
| .00	      	| Priority  					 				|
| .00				  | End of No Passing 3.5 tons  |
