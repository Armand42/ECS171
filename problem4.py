#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  6 13:27:29 2019

@author: armandnasserischool
"""

# Need to run for 1 epoch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings("ignore")
# Setting Seed and Importing the Dataset
np.random.seed(0)
dataset = pd.read_csv('yeast.data', delim_whitespace=True, 
                       names=["Sequence Name","MCG","GVH","ALM","MIT","ERL","POX","VAC","NUC","Class Distribution"])


X = np.array([[0.58,0.61,0.47,0.13,0.50,0,0.48,0.22]])
y = np.array([[0,0,0,0,0,0,1,0,0,0]])

# Initialising the ANN
classifier = Sequential()
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 3, activation = 'sigmoid', input_dim = 8))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 3, activation = 'sigmoid'))
# Adding the output layer
classifier.add(Dense(output_dim = 10, activation = 'sigmoid'))

# Extracting all the weights
hiddenWt1 = classifier.layers[0].get_weights()
hiddenWt2 = classifier.layers[1].get_weights()
outputWt = classifier.layers[2].get_weights()


hiddenWt1[0][:,0] = 1
hiddenWt1[1][0] = 1

hiddenWt2[0][:,0] = 1
hiddenWt2[1][0] = 1

outputWt[0][:,0] = 1
outputWt[1][0] = 1

classifier.layers[0].set_weights(hiddenWt1)
classifier.layers[1].set_weights(hiddenWt2)
classifier.layers[2].set_weights(outputWt)

# Compiling the ANN
opt = keras.optimizers.SGD(lr = 0.01)
classifier.compile(optimizer = opt, loss = 'mean_squared_error', metrics = ['accuracy'])

classifier.fit(X, y, batch_size = 1, nb_epoch = 1, verbose=1)

# Printing all the weights per layer
print("Bias: ")
print("Bias Hidden Layer 1:",classifier.layers[1].get_weights()[1][0])
print("Bias Output Layer:",classifier.layers[2].get_weights()[1][0])
print("Hidden Layer Weights: ")
print("wt1HL:",classifier.layers[1].get_weights()[0][:,0][0])
print("wt2HL:",classifier.layers[1].get_weights()[0][:,0][1])
print("wt3HL:",classifier.layers[1].get_weights()[0][:,0][2])
print("Output Layer Weights: ")
print("wt1OL:",classifier.layers[2].get_weights()[0][:,0][0])
print("wt2OL:",classifier.layers[2].get_weights()[0][:,0][1])
print("wt3OL:",classifier.layers[2].get_weights()[0][:,0][2])