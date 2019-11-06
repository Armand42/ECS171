#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:48:53 2019

@author: armandnasserischool
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import IsolationForest
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

# Using the isolation forest algorithm to remove the outliers
isol = IsolationForest(random_state = 0)
isol.fit(X)
# Removing all the outliers
# Prints the new dataset of removed outliers 
index = 0
num_out = 0
for i in isol.predict(X):
    if (i < 0 and index < len(X)):
        num_out += 1
        X = X.drop(X.index[index])
        y = y.drop(y.index[index])
    index = index + 1

# Binary encoding the y column into 10 binary encoded columns
y = pd.get_dummies(y)

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.34, random_state = 0)

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
history = classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 300, verbose=1, callbacks = [weightCall], validation_data = (X_test,y_test))
# Calculating the training and test loss per iteration
test_loss = [1-x for x in history.history['val_acc']]
train_loss = [1-x for x in history.history['acc']]

# Plotting the Weights per iteration of the last layer
plt.title("Weights per iteration of last layer")
plt.xlabel("Epochs")
plt.ylabel("Weights")
plt.plot(bias, label = "bias")
plt.plot(w1, label = "w1")
plt.plot(w2, label = "w2")
plt.plot(w3, label = "w3")
plt.legend()
plt.show()
# Plotting the Training and Testing Error per iteration
plt.title("Training & Testing Error per Iteration")
plt.plot(train_loss, label = "Train Loss", color = "green")
plt.plot(test_loss, label = "Test Loss", color = "blue")
plt.xlabel("Epochs")
plt.ylabel("Ratio")
plt.legend()
plt.show()



