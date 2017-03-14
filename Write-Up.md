#**Traffic Sign Recognition** 

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: http://i.imgur.com/z3bgoiP.png "Visualization"
[image2]: http://i.imgur.com/up8hmsG.png "Pre-Processing"
[image3]: http://i.imgur.com/rOzEZst.png "6 Raw Signs"
[image4]: http://i.imgur.com/7sKXj3U.png "soft5"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/xmfcx/Udacity-P2-Traffic-Sign-Classifier/blob/master/Traffic_Sign_Classifier.ipynb)

###Data Set Summary & Exploration

####1. Provide a basic summary of the data set and identify where in your code the summary was done. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

The code for this step is contained in IPython notebook cell 2 to 4.  

I used the basic python functionality to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32)
* The number of unique classes/labels in the data set is 43

####2. Include an exploratory visualization of the dataset and identify where the code is in your code file.

The code for this step is contained in IPython notebook cell 5 to 6.  

Here is an exploratory visualization of the data set. It is a bar chart showing how the data is distributed.

![alt text][image1]

###Design and Test a Model Architecture

####1. Describe how, and identify where in your code, you preprocessed the image data. What tecniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc.

The code for this step is contained in IPython notebook cell 7 to 11.  

As a first step, I decided to convert the images to grayscale because it brings simplicity and faster training time, smaller network, uniformity at cost of color info.
Then applied mean substraction to center data around origin.
Then normalized the image by dividing each dimension by its deviation so images are treated equally.

Here is an example of a traffic sign image before and after pre-processing.

![alt text][image2]

####2. Describe how, and identify where in your code, you set up training, validation and testing data. How much data was in each set? Explain what techniques were used to split the data into these sets. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, identify where in your code, and provide example images of the additional data)

The code for splitting the data into training and validation sets is contained in IPython notebook cell 11. 
Splitted with pandas train_test_split method in 1 to 4 ratio.
My final training set had 27839 number of images. My validation set and test set had 6960 and 12630 number of images.

####3. Describe, and identify where in your code, what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

The code for my final model is located in ipython notebook cell 13. 

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 Grayscale image   							| 
| Layer 1) Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 					|
| Dropout	    	  	| 0.9 keeping 									|
| Layer 2) Convolution 5x5     	| 1x1 stride, valid padding, outputs 10x10x16 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 					|
| Dropout	    	  	| 0.9 keeping 									|
| Flatten				|	outputs 5x5x16=400							|
| Layer 3) Fully connected		| outputs 120     						|
| RELU					|												|
| Dropout	    	  	| 0.9 keeping 									|
| Layer 4) Fully connected		| outputs 84     						|
| RELU					|												|
| Dropout	    	  	| 0.9 keeping 									|
| Layer 5) Fully connected		| outputs 43     						|
| Softmax				| 		      									|
 

####4. Describe how, and identify where in your code, you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

The code for training the model is located in ipython notebook cell 15 to 18. 

EPOCHS = 80
BATCH_SIZE = 128
dropout_ratio = 0.9
optimizer = adam optimizer
learning rate = 0.001

####5. Describe the approach taken for finding a solution. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

The code for calculating the accuracy of the model is located in ipython notebook cell 18. 


My final model results were:
* validation set accuracy of 0.983 
* test set accuracy of 0.926

If an iterative approach was chosen:

* What was the first architecture that was tried and why was it chosen?

I moderately tweaked the well known LeNet architecture. It was originally used for classifying hand written digits, and it is similar to what we are trying to achieve here. That's why it is suitable.

* What were some problems with the initial architecture?

Its initial output class amount didn't match ours.

* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to over fitting or under fitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.

Its output classes were not 34, it didn't preprocess like I did, it didn't have dropout.

* Which parameters were tuned? How were they adjusted and why?

It was a little bit overfitting because it was having high accuracy on training but slightly lower accuracy on test data. So I added dropout layers at multiple stages to minimize overfitting.

* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

Convolution layers help the network to hierarchically learn different levels of features. Dropout helps making network lose the unnecessary ineffective nodes which actually causes overfitting by randomly dropping and testing them iteratively.

If a well known architecture was chosen:

* What architecture was chosen?

LeNet

* Why did you believe it would be relevant to the traffic sign application?

It was originally used for classifying hand written digits, and it is similar to what we are trying to achieve here. That's why it is suitable.

* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?

Network was able to classify 92.6% of 12630 traffic signs it had never seen before correctly. It is strong. If I applied data augmentation by stretching and rotating the initial dataset it would even be higher than this. But still, quite strong since it is sort of universal and not only bound to traffic sign domain. 

###Test a Model on New Images

####1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![][image3]

The first image might be difficult to classify because it is a bit rotated. Rest should be clear enough.

####2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. Identify where in your code predictions were made. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

The code for making predictions on my final model is located in Ipython notebook cell 26.

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Rotated bicycle      		| Children crossing   									| 
| stop    			| Stop 										|
| pedestrians    			| Speed limit (70km/h) 										|
| turn-right					| Turn right ahead											|
| bicycle	      		| End of all speed and passing limits				 				|
| road-work			| Road work    							|


The model was able to correctly guess 3 of the 6 traffic signs, which gives an accuracy of 50%. This compares not really favorably to the accuracy on the test set of 0.926. Maybe the signs I have chosen weren't germanic enough like on the dataset. It's something.

####3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction and identify where in your code softmax probabilities were outputted. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 29th cell of the Ipython notebook.

![][image4]

Except for bicycle2 where it has 0.8, 0.2 indecisiveness, model is more than 99% sure about what it predicts. I am not sure what makes it so sure about what it predicts.
