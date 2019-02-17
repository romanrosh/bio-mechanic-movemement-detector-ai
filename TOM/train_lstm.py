from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import LSTM
from keras.utils import to_categorical
from matplotlib import pyplot
from keras.preprocessing.sequence import pad_sequences
import os
import numpy as np

CURRENT_DIR = os.getcwd()
TARGET_CSV = os.path.join(CURRENT_DIR, './nps/')


def load_data(folder):
    X1 = np.load(folder + '1_X.npy')
    y1 = np.load(folder + '1_y.npy')
    X0 = np.load(folder + '0_X.npy')
    y0 = np.load(folder + '0_y.npy')
    return X1, y1, X0, y0


def padding(array_0, array_1):
    maxlen = max(max(map(lambda x: x.shape[0], array_0)), max(map(lambda x: x.shape[0], array_1)))
    padded0 = pad_sequences(array_0, padding='post', maxlen=maxlen)
    padded1 = pad_sequences(array_1, padding='post', maxlen=maxlen)
    X = np.vstack([padded1, padded0])
    return X


def evaluate_model(trainX, trainy, testX, testy):
    verbose, epochs, batch_size = 0, 15, 16
    n_timesteps, n_features, n_outputs = trainX.shape[1], trainX.shape[2], trainy.shape[1]
    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    _, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)
    return accuracy


if __name__ == '__main__':
    X1, y1, X0, y0 = load_data(TARGET_CSV)
    X = padding(X0, X1)
    y = np.hstack([y0, y1])
    y = to_categorical(y)
    accuracy = evaluate_model(X, y, X, y)
    print(accuracy)
