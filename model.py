# import libraries

from scipy import ndimage
import numpy as np
import csv 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

from keras.models import Sequential
from keras.layers import Flatten , Dense , Conv2D , MaxPooling2D , Activation , Cropping2D , Lambda , Dropout


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

# plot sample images 

def plot_samples(samples , label) :
	'''
	This function plots sample images from the data set
	label argument determines whether 'center' or 'center and left and right' or 'center and flipped'
	images are visulaized
	''' 

	if label == 'center' :
		f , axes = plt.subplots(3,3 , figsize=(10,6))
		for i in range(9):
			idx = np.random.randint(len(samples))
			axes = axes.ravel()
			source_path = './data/IMG/'
			image_path = source_path + train_samples[idx][0].split('/')[-1]
			image = mpimg.imread(image_path)
			axes[i].imshow(image)
			axes[i].axis('off')
			axes[i].set_title('center')
		f.subplots_adjust(hspace = 0.001 , wspace= 0.2)
		f.savefig('./examples/center_images.png')

	elif label == 'all' :
		f , axes = plt.subplots(3,3 , figsize=(10,6))
		for i in range(3):
			idx = np.random.randint(len(samples))
			source_path = './data/IMG/'
			titles = ['center' , 'left' , 'right']
			for j in range(3) :
				image_path = source_path + samples[idx][j].split('/')[-1]
				image = mpimg.imread(image_path)
				axes[i][j].imshow(image)
				axes[i][j].axis('off')
				axes[i][j].set_title(titles[j])
		f.subplots_adjust(hspace = 0.001 , wspace= 0.2)
		f.savefig('./examples/camera_images.png')

	elif label == 'flip' :
		f , axes = plt.subplots(3,2 , figsize=(10,6))
		for i in range(3):
			idx = np.random.randint(len(samples))
			source_path = './data/IMG/'
			for j in range(2) :
				image_path = source_path + samples[idx][0].split('/')[-1]
				image = mpimg.imread(image_path)
				flipped_image = np.fliplr(image)
				img = flipped_image if (j+1)%2 == 0 else image
				title = 'flipped' if (j+1)%2 == 0 else 'original'
				axes[i][j].imshow(img)
				axes[i][j].axis('off')
				axes[i][j].set_title(title)
		f.subplots_adjust(hspace = 0.001 , wspace= 0.2)
		f.savefig('./examples/flipped_images.png')

		

	plt.show()


plot_samples(train_samples , 'center')
plot_samples(train_samples , 'all')
plot_samples(train_samples , 'flip')

# create data generator

def generator( samples , batch_size=32 ) :
	num_samples = len(samples)
	
	while 1: # Loop forever so the generator never terminates
		shuffle(samples)
		for offset in range(0 , num_samples , batch_size) :
			batch_samples = samples[offset : offset + batch_size]

			correction=0.15
			
			images = []
			angles = []

			for batch_sample in batch_samples :
				source_path = './data/IMG/'
				center_image = ndimage.imread(source_path+batch_sample[0].split('/')[-1])
				left_image   = ndimage.imread(source_path+batch_sample[1].split('/')[-1])
				right_image  = ndimage.imread(source_path+batch_sample[2].split('/')[-1])
				
				center_angle = float(batch_sample[3])
				left_angle   = center_angle + correction
				right_angle  = center_angle - correction

				flipped_image  = np.fliplr(center_image)
				flipped_angle = -center_angle
				
				images.extend([ center_image , left_image , right_image , flipped_image ])
				angles.extend([ center_angle , left_angle , right_angle , flipped_angle ])

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

model.add(Conv2D(filters=64 , kernel_size=3 , strides=1 , padding='valid'))
model.add(Activation('relu'))
model.add(Conv2D(filters=64 , kernel_size=3 , strides=1 , padding='valid'))
model.add(Activation('relu'))

# fully_connected layers

model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(units=100 , activation='relu'))
model.add(Dense(units=50 , activation='relu'))
model.add(Dense(units=10 , activation='relu'))
model.add(Dense(units=1))

# configure the training process

model.compile(loss='mse' , optimizer='adam')

# summarize the model

model.summary()

# train the model

history_object = model.fit_generator(train_generator , steps_per_epoch=len(train_samples)/64 , epochs=5 , validation_data = validation_generator , validation_steps=len(validation_samples)/64 )


# loss visualization

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('mean squared error loss')
plt.title('loss during training')
plt.legend(['train_loss' , 'valid_loss'] , loc='upper right')
plt.savefig('visualizing_loss.png')

# save the model

model.save('model.h5')

print('model saved !')
