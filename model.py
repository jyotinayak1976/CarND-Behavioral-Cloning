import csv
import cv2
import numpy as np


lines = []
# The csv file is read here.
with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
lines = lines[1:]

# Here images from the center camera is loaded.
for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)

# Here images from the left camera is added.    
for line in lines:
    source_path = line[1]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    # adjustment of 0.26 is made for the steering angle here to adust for images taken from left image.
    measurement = float(line[3]) + 0.26
    measurements.append(measurement)
    
for line in lines:
    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    # adjustment of -0.26 is made for the steering angle here to adust for images taken from right image.
    measurement = float(line[3]) - 0.26
    measurements.append(measurement)

# Now with 0.5 probability images are picked from the entire list of images and flipped.
# The sign of the steering angle is also reversed by multiplying -1.
flipped_images = []  
flipped_measurements = [] 
for image,measurement in zip(images,measurements):  
    flip = np.random.randint(2)
    if flip == 1:
        flipped_image = cv2.flip(image,1)
        flipped_images.append(flipped_image)
        flipped_measurements.append((-1) * measurement)
# Flipped images and measurements are combined to the earlier list of images and measurement here.        
images = images + flipped_images
measurements = measurements + flipped_measurements

X_train = np.array(images)
y_train = np.array(measurements)

# Inclued all the keras library for model buidling.
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape=(160,320,3)))
# croppling layer is added to crop the images here.
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Convolution2D(24, 5, 5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(36, 5, 5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(48, 5, 5, subsample=(2,2),activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Convolution2D(64, 3, 3, activation='relu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer = 'adam')
model.fit(X_train, y_train, validation_split = 0.3, shuffle = True, nb_epoch = 7)

#Here model is saved.
model.save('model.h5')

