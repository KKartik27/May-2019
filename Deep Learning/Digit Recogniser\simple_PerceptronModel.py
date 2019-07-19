# -*- coding: utf-8 -*-
"""
@author: Sreenivas.J
"""
import os
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils

os.chdir("D:/Data Science/Data")
np.random.seed(100)
    
digit_train = pd.read_csv("Digits Recognizer_Train.csv")
digit_train.shape
digit_train.info()

#iloc[:, 1:] Means first to last row and 2nd column to last column
#255.0 --> Convert my data to 255 pixels
X_train = digit_train.iloc[:, 1:]/255.0
y_train = np_utils.to_categorical(digit_train["label"])
y_train.shape

#Here comes the Basic Neural Network
model = Sequential()
model.add(Dense(10, input_shape=(784,), activation='softmax'))
print(model.summary())

#mean_squared_error for regression
#Binary_Crossentropy is for Binary Classification and categorical_crossentropy is for multi class classification
#Both uses the log loss
#In SGD, before for-looping, system randomly shuffle the training examples.
#For further reading regarding Cross entropy:
#Here is a good article: https://www.quora.com/Whats-an-intuitive-way-to-think-of-cross-entropy
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])

#Batch Size — The number of training examples in one forward/backward pass.
#Each Epoch takes entire train data
history = model.fit(x=X_train, y=y_train, verbose=1, epochs=4, batch_size=2, validation_split=0.2)
print(model.get_weights())




