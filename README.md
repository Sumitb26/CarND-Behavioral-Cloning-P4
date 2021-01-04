# **Behavioral Cloning** 

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./train_data_map1/IMG/center_2020_12_21_15_58_05_390.jpg "Normal Image"
[image2]: ./examples/flipped_center_2020_12_21_15_58_05_390.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network
* README.md summarizing the results

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model consists of a convolution neural network with 5x5 filter sizes and depths between 25 and 64 (model.py lines 36-40)

The model includes RELU activation to introduce nonlinearity, and the data is normalized in the model using a Keras lambda layer (code line 34).

#### 2. Attempts to reduce overfitting in the model

The model was trained and validated on different data sets and data augmentation was used to ensure that the model was not overfitting. The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 47).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road and flipping the data i.e driving car in opposite direction.

For details about how I created the training data, see the next section.

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

My first step was to use a convolution neural network model similar to the LeNet I thought this model might be appropriate because ConvNet works with wide range on input image sizes.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting.

To combat the overfitting, I modified the model by augmenting training data by flipping images and using left and right camera images.

Then I cropped the top portion of image as it contained useless information about trees and sky. This results in faster training as each image is focused on only that part of image that is useful for predicting steering angle.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track to improve the driving behavior in these cases, gathering more data fixed the issue.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 33-45) consisted of a convolution neural network with the following layers and layer sizes:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 160x320x3 RGB image  							| 
| Cropping2D         	| outputs 70x25                                	|
| Conv2D				| 2x2 stride, RELU activation, filters 25   	|
| Conv2D				| 2x2 stride, RELU activation, filters 36   	|
| Conv2D				| 2x2 stride, RELU activation, filters 48   	|
| Conv2D				| 1x1 stride, RELU activation, filters 64   	|
| Conv2D				| 1x1 stride, RELU activation, filters 64   	|
| Flatten				|                       						|
| Dense         		| outputs 100            						|
| Dense         		| outputs 50            						|
| Dense         		| outputs 10            						|
| Dense         		| outputs 1              						|


#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded a lap on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image1]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to adjust oro get back on track if it deviates.

To augment the data sat, I also flipped images and angles thinking that this would generate more training data which helps the model in generalizing and avoid overfitting. For example, here is an image that has then been flipped:

![alt text][image1]
![alt text][image2]

After the collection process, I had X number of data points. I then preprocessed this data using lambda layer which helped to parallelize image normalization.


I finally randomly shuffled the data set and put 20% of the data into a validation set.

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 5 as evidenced by increased validation loss after 5th epoch. I used an adam optimizer so that manually training the learning rate wasn't necessary.
