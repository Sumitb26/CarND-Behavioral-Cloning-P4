import csv
import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from PIL import Image

car_images = []
steering_angles = []
with open('train_data_map1/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        steering_center = float(row[3])
        correction = 0.2 # this is a parameter to tune
        steering_left = steering_center + correction
        steering_right = steering_center - correction

        # read in images from center, left and right cameras
        img_center = cv2.imread(row[0])
        img_left = cv2.imread(row[1])
        img_right = cv2.imread(row[2])
        # add images and angles to data set
        car_images.extend((img_center, img_left, img_right, np.fliplr(img_center), np.fliplr(img_left), np.fliplr(img_right)))
        steering_angles.extend((steering_center, steering_left, steering_right, -steering_center, -steering_left, -steering_right))

X_train = np.array(car_images)
y_train = np.array(steering_angles)
print(len(X_train))
print(len(y_train))

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(25,(5,5),strides=(2,2), activation="relu"))
model.add(Conv2D(36,(5,5),strides=(2,2), activation="relu"))
model.add(Conv2D(48,(5,5),strides=(2,2), activation="relu"))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(Conv2D(64,(3,3), activation="relu"))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=5)

model.save('model.h5')
