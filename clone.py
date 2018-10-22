# import libraries

from scipy import ndimage
import numpy as np
import os 
import csv 
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten , Dense , Conv2D , MaxPooling2D , Activation , Cropping2D , Lambda


# parse driving_log.csv 

samples = []
with open("./data/driving_log.csv") as csv_file :
	reader = csv.reader(csv_file)
	for line in reader :
		# exclude column labels
		if line == ['center', 'left', 'right', 'steering', 'throttle', 'brake', 'speed'] :
			continue
		samples.append(line)

# divide samples into train_samples and validation_samples

train_samples , validation_samples = train_test_split(samples , test_size = 0.2)

# create data generator

def generator( samples , batch_size=32 ) :
	num_samples = len(samples)
	
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0 , num_samples , batch_size) :
			batch_samples = samples[offset : offset + batch_size]

			images = []
			angles = []
			for batch_sample in batch_samples :
				source_path = './data/IMG/'
				current_path = source_path+batch_sample[0].split('/')[-1]
				center_image = ndimage.imread(current_path)
				center_angle = float(batch_sample[3])
				flipped_image  = np.fliplr(center_image)
				reversed_angle = -center_angle
				images.append(center_image)
				images.append(flipped_image)
				angles.append(center_angle)
				angles.append(reversed_angle)

			X_data = np.array(images)
			y_data = np.array(angles)

			yield shuffle(X_data,y_data)


train_generator = generator(train_samples , batch_size = 64)
validation_generator = generator(validation_samples , batch_size = 64)

# build model architecture

model = Sequential()

# data pre-processing layers

model.add(Cropping2D(cropping=((50,20),(0,0)) , input_shape=(160,320,3)))    # cropping to focus on the road section 
model.add(Lambda(lambda x : (x/255.0) - 0.5 ))                                # normalization

# convolutional layers

model.add(Conv2D(filters=24 , kernel_size=5 , strides=2 , padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(filters=36 , kernel_size=5 , strides=2 , padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(filters=48 , kernel_size=5 , strides=2 , padding='valid'))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=2 , strides=2 , padding='valid'))
model.add(Conv2D(filters=64 , kernel_size=3 , strides=1 , padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(filters=64 , kernel_size=3 , strides=1 , padding='valid'))
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=2 , strides=2 , padding='valid'))

# fully_connected layers

model.add(Flatten())
model.add(Dense(units=100 , activation='relu'))
model.add(Dense(units=50 , activation='relu'))
model.add(Dense(units=10 , activation='relu'))
model.add(Dense(units=1))

# configure the training process

model.compile(loss='mse' , optimizer='adam')

# train the model

model.fit_generator(train_generator , steps_per_epoch=len(train_samples)/64 , epochs=5 , validation_data = validation_generator , validation_steps=len(validation_samples)/64 )

#model.fit_generator(train_generator, steps_per_epoch= len(train_samples), 
#validation_data=validation_generator, validation_steps=len(validation_samples), epochs=5, verbose = 1)
# save the model

model.save('model.h5')

print('model saved !')
