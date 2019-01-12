
# Load util
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

from keras.models import Sequential, Model
from keras import optimizers
from keras.layers import Dense, Activation, Conv2D, MaxPool2D, Flatten, BatchNormalization, Dropout
from keras.preprocessing.image import ImageDataGenerator
df=pd.read_csv('C:/Users/romanrosh/dataframe.csv')


batch_size = 60
from keras.preprocessing.image import ImageDataGenerator

X = df.loc[:,:'label']
Y = df.loc[:,'label']

channels = 1

"""## Convolution Neural Networks (CNN)"""

cnn = Sequential()
cnn.add(Conv2D(32, kernel_size=(3, 3)))
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

cnn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
cnn.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10)

print(score)
