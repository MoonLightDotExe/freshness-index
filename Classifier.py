import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten
from keras.models import Sequential
import cv2

import os

path = './Resources'

data_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    fill_mode="nearest",
    horizontal_flip=True)

data = data_gen.flow_from_directory(
    path,
    target_size=(256, 256),
    batch_size=32,
    class_mode="categorical"
)
model = Sequential()
model.add(Conv2D(256, (3, 3),activation="relu", input_shape=(256, 256, 3)))
model.add(MaxPool2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPool2D((2, 2)))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPool2D((2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPool2D((2, 2)))

model.add(Flatten())

model.add(Dense(32, activation="relu"))
model.add(Dense(16, activation="relu"))
model.add(Dense(16, activation="relu"))

model.add(Dense(6, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy", "mae"])

model.summary()

model.fit(data, batch_size=32, epochs=5)

model.save('classifier_model2.keras')

