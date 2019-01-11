
# Load util
import matplotlib.pyplot as plt

import numpy as np
import glob

from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator

dataset_folder_path = 'C:/Users/romanrosh/photos'
train_folder = dataset_folder_path + '/side'
test_folder = dataset_folder_path + '/side'

test_files = glob.glob(test_folder + '/**/*.jpg')
train_files = glob.glob(train_folder + '/**/*.jpg')

train_examples = len(train_files)
test_examples = len(test_files)
print("Number of train examples: ", train_examples)
print("Number of test examples: ", test_examples)
batch_size = 60
from keras.preprocessing.image import ImageDataGenerator

"""View some sample images:"""

datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=5,
    zoom_range=0.2,
    horizontal_flip=True)

img_height = img_width = 200
channels = 1
if (channels == 1):
    color_mode_ = "grayscale"
else:
    color_mode_ = "rgb"

train_generator = datagen.flow_from_directory(
    train_folder,
    color_mode=color_mode_,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    class_mode='binary'
)

test_generator = datagen.flow_from_directory(
    test_folder,
    color_mode=color_mode_,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True,
    class_mode='binary'
)

"""## Convolution Neural Networks (CNN)"""
print(test_files)

cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(3, 3), input_shape=(img_height, img_width, 1)))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))

cnn.add(MaxPool2D(pool_size=(2, 2)))

cnn.add(Conv2D(64, kernel_size=(3, 3)))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))

cnn.add(MaxPool2D(pool_size=(2, 2)))

cnn.add(Conv2D(128, kernel_size=(3, 3)))
cnn.add(BatchNormalization())
cnn.add(Activation('relu'))

cnn.add(MaxPool2D(pool_size=(2, 2)))

cnn.add(Flatten())

cnn.add(Dense(256, activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.5))

cnn.add(Dense(128, activation='relu'))
cnn.add(BatchNormalization())
cnn.add(Dropout(0.5))

cnn.add(Dense(32, activation='relu'))
cnn.add(BatchNormalization())

cnn.add(Dense(1, activation='softmax'))

cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
cnn.summary()

history_cnn = cnn.fit_generator(train_generator, train_examples // batch_size, verbose=1, epochs=15)
score = cnn.evaluate_generator(test_generator, test_examples // batch_size, verbose=1)
print(score)
