# **Behavioral Cloning for self-driving cars** 

In this project, A convolutional neural network is used to clone driving behavior. The model is trained, validated and tested using Keras. The model will output a steering angle to an autonomous vehicle.

---

**The goals / steps of this project are the following:**
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[nvidia]: ./examples/cnn_architecture.png  "Model Visualization"
[model]: ./examples/model_summary.png      "Model layers"
[center]: ./examples/center_images.png     "center camera images"
[l_r]:    ./examples/camera_images.png     "camera images"
[flip]: ./examples/flipped_images.png "flipped images"


---

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted 

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

**This github repository includes the following files:**
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup.md` summarizing the results
* `autonomous_driving.mp4` which is a video recording of the vehicle driving autonomously around the track for one full lap


#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed
The data is cropped to focus on the road section because the car hood and the environment above the horizon may be a distraction for the model then normalized in the model using a Keras `Cropping2D` and `Lambda` layers (clone.py lines 76-77).
This is a regression problem as the steering command should be predicted from the road images captured by the front-facing camera.I implemented the nvidia architectire published in this paper [End to End Learning for Self-Driving Cars](https://arxiv.org/pdf/1604.07316.pdf). 
The model consists of 3 convolution layers with 5x5 kernel sizes , strides of 2 and features maps of 24 , 36 , 48  and `valid` padding (clone.py lines 81-86) , followed by 3 convolutional layers with 3x3 kernel sizes , strides of 1 and features maps of 64 , 64 and `valid` padding (clone.py lines 88-91).The output of the last convolutional layer is flattened and fed into 3 fully-connected layers with hidden units 100 , 50 , 10 and finally a single output node (clone.py lines 95-99).The model includes RELU layers to introduce nonlinearity.

![alt text][nvidia] 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (clonel.py line 96). 
The model was trained and validated on different data sets to ensure that the model was not overfitting (clone.py line 107). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer with the default `learning rate = 0.001` . Dropout was used in order to avoid overfitting and after some fine-tuning , it as set to 0.2 which achieved the best performance

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road back to the center by making use of multiple camera approach mentioned in the paper. This is clarification made by udacity that helps understand it.

>In the simulator, you can weave all over the road and turn recording on and off to record recovery driving. 
>In a real car, however, that’s not really possible. At least not legally.
>so a real car, we’ll have multiple cameras on the vehicle, and we’ll map recovery paths from each camera. For example, if you train the model to associate a given image from the center camera with a left turn, then you could also train the model to associate the corresponding image from the left camera with a somewhat softer left turn. And you could train the model to associate the corresponding image from the right camera with an even harder left turn.
>In that way, you can simulate your vehicle being in different positions.
>From the perspective of the left camera, the steering angle would be less than the steering angle from the center camera. From the right camera's perspective, the steering angle would be larger than the angle from the center camera.

---

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to build a simple model to verify that it can be designed in keras , be trained and validated on the camera road images then used to drive the car in autonomous mode in the car simulator.

My first step was to use a LeNet5 architecture because it's was used in an image recognition task . I thought this model might be a starting point.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set as it helps measure how well the model generalizes to useen samples. I found that my first model had a high mean squared error on the training set and a high mean squared error on the validation set. This implied that the model was underfitting. 

To combat the underfitting, I modified the model so that It can achieve better peformance on both the training and validation sets by preceding the convolutional layers by data processing layers.A cropping layer was added into the begining of the model to crop the images from top and bottom and eliminate the portions of the scene that may distract the model in order to focus on the road section which contains more beneficial features to predict the steering angle .After that , A normalization layer was added to normalize the image pixel values to `[-0.5 , 0.5]` as it helps the training converge faster. The performance was better. I decided to use the center camera images as well as flipping them and reversing the associated angle to remove the bias in the training set towards left turns. Then I implemented the nvidia architecture which is depicted in the above figure.the training loss was 0.0086 and the validation loss with 0.0094. There are different approachs to teach the car how to recover from the sides of the road back to the center. I decided to make use of the left and right camera images by adding a correction factor to the associated steering angle as the following:    <br>  
`left_angle  = associated center angle + correction factor`  to train the network to steer a little harder to the right<br>
`right_angle = associated center angle - correction factor`  to train the network to steer a little harder to the left

Those additional images are incorported in the training data and the model was trained on it to achieve training loss of 0.0162 and validation loss of 0.0207 at a correction factor 0f `0.3`. The correction factor was fine-tuned until it reached `0.15` . To avoid overfitting a dropout layer was used after the flatten layer with dropout probability of `0.2`.
the model was compiled using Adam optimizer and mean squred error as loss then it was trained and validated on training and validation sets using 5 epochs .The model was saved and the `model.h5' file was produced ,  which stores the network weights .  
The final step was to run the simulator to see how well the car was driving around track one. It completed track one successfully without leaving the drivable portion of the track surface.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture is located in (clone.py lines 70-100) and here is a visualization of the architecture.

![alt text][model]

**Total params: 981,819**   
**Trainable params: 981,819**   
**Non-trainable params: 0**   

#### 3. Creation of the Training Set & Training Process

I used [the sample driving data](https://d17h27t6h515a5.cloudfront.net/topher/2016/December/584f6edd_data/data.zip) provided by udacity. It records several driving laps through track one.

![alt text][center]

I also used left and right camera images along with center images to collect more training data and to make the model learn the recovery process .

![alt text][l_r]

To augment the data set, I also flipped images and angles thinking that this would remove the bias towards left turns because most turns in track one are left turns.  
 For example, here is an image that has then been flipped


![alt text][flip]


After the collection process, I had 30000 number of data points.

I finally randomly shuffled the data set and put 20% of the data into a validation data.
I used data generators for training set and validation set to reduce memory consumption. 
I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The number of epochs was 5. The batch size was 64. The loss function used was mean squared error because it suits the regression problem. I used an adam optimizer so that manually training the learning rate wasn't necessary.
