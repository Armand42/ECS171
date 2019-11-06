#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:48:53 2019

@author: armandnasserischool
"""

import numpy as np
import pandas as pd
import keras
from keras.models import Sequential
from keras.layers import Dense
import warnings
warnings.filterwarnings("ignore")
# Setting Seed and Importing the Dataset
np.random.seed(0)
dataset = pd.read_csv('yeast.data', delim_whitespace=True, 
                       names=["Sequence Name","MCG","GVH","ALM","MIT","ERL","POX","VAC","NUC","Class Distribution"])
# Global weight and bias arrays
extractedCYT = []
bias = []
w1 = []
w2 = []
w3 = []
# A callback class function that calculates the weights for train/test error per iteration
class weightsCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lastLayer = self.model.layers[len(self.model.layers) - 1]
        weights = lastLayer.get_weights()[0]
        bias_sing = lastLayer.get_weights()[1]
        # Append weights and bias to global arrays
        w1.append(weights[0][0])
        w2.append(weights[1][0])
        w3.append(weights[2][0])
        bias.append(bias_sing[0])
               
# Instantiating the callback function       
weightCall = weightsCallback()
# Splitting dataset between X and y so that y can be encoded
dataset = dataset.drop(columns = ['Sequence Name'])
y = dataset['Class Distribution']
X = dataset.iloc[:,:].values
X = dataset.drop(columns = 'Class Distribution')

# Binary encoding the y column into 10 binary encoded columns
y = pd.get_dummies(y)

# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 3, activation = 'sigmoid', input_dim = 8))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 3, activation = 'sigmoid'))
# Adding the output layer
classifier.add(Dense(output_dim = 10, activation = 'sigmoid'))
# Compiling the ANN
classifier.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['accuracy'])

# Fitting the ANN to the Training set and calling the callback function to calculate the weights per iteration
history = classifier.fit(X, y, batch_size = 10, nb_epoch = 100, verbose=1, callbacks = [weightCall], validation_data = (X,y))
# Calculating the training loss per iteration
train_loss = [1-x for x in history.history['acc']]

# Printing out the final hidden layer weights, bias, and training loss
print("Final Hidden Layer Weights")
print("bias:",bias[-1])
print("w1:",w1[-1])
print("w2:",w2[-1])
print("w3:",w3[-1])
print("Train Loss:",train_loss[-1])


