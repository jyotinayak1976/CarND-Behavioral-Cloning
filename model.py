import csv
import cv2
import numpy as np


lines = []

with open('data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

images = []
measurements = []
lines = lines[1:]

for line in lines:
    source_path = line[0]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3])
    measurements.append(measurement)
    
for line in lines:
    source_path = line[1]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3]) + 0.26
    measurements.append(measurement)
    
for line in lines:
    source_path = line[2]
    filename = source_path.split('/')[-1]
    current_path = 'data/IMG/' + filename
    image = cv2.imread(current_path)
    images.append(image)
    measurement = float(line[3]) - 0.26
    measurements.append(measurement)
    
flipped_images = []  
flipped_measurements = [] 
for image,measurement in zip(images,measurements):  
    flip = np.random.randint(2)
    if flip == 1:
        flipped_image = cv2.flip(image,1)
        flipped_images.append(flipped_image)
        flipped_measurements.append((-1) * measurement)
        
images = images + flipped_images
measurements = measurements + flipped_measurements

X_train = np.array(images)
y_train = np.array(measurements)

print('X_train shape', X_train.shape)
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
model.add(Lambda(lambda x:x/255.0 - 0.5, input_shape=(160,320,3)))
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

model.save('model.h5')

