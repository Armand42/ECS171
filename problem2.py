#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:48:53 2019

@author: armandnasserischool
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing
import keras
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle
from sklearn.ensemble import IsolationForest
from keras.callbacks import LambdaCallback
import warnings
warnings.filterwarnings("ignore")

# NEED ONE HOT ENCODER

dataset = pd.read_csv('yeast.data', delim_whitespace=True, 
                       names=["Sequence Name","MCG","GVH","ALM","MIT","ERL","POX","VAC","NUC","Class Distribution"])

extractedCYT = []
bias = []
w1 = []
w2 = []
w3 = []
class weightsCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        lastLayer = self.model.layers[len(self.model.layers) - 1]
        weights = lastLayer.get_weights()[0]
        bias_sing = lastLayer.get_weights()[1]
        weightnow = []
        w1.append(weights[0][0])
        w2.append(weights[1][0])
        w3.append(weights[2][0])
        bias.append(bias_sing[0])
       # print(weights)
        #for weight in weights:
            #print(weight,i)
            #print('\n\n\n\nweight: ')
            #print(weight)
         #   w1.append(weight[0])
          #  w2.append(weight[1])
           # w2.append(weight[2])
            
            
            #print(extractedCYT)
        
        #extractedCYT.append([bias[0]] + weightnow)
        
      
        
weightCall = weightsCallback()
#dataset = dataset.sample(n=1484)
# X features dropping Sequence Column
dataset = dataset.drop(columns = ['Sequence Name'])
y = dataset['Class Distribution']
X = dataset.iloc[:,:].values
X = dataset.drop(columns = 'Class Distribution')


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

#print(X.shape)
#print(y.shape)


#print(X)
# Encoding categorical data
#labelencoder = LabelEncoder()
#y = labelencoder.fit_transform(y)
#print(y)

#print(y)
# One hot encoder to binarization
#onehotencoder = OneHotEncoder(categorical_features = 10)
y = pd.get_dummies(y)
#print(y)
#print(y.shape)
#y = onehotencoder.fit_transform(y).toarray()

# Input dimensions for new encoded data
#newX = X[:,10:18]
#newY = X[:,0:10]

#print(X.shape)


# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.34, random_state = 0)


# Initialising the ANN
classifier = Sequential()
# Drop Sequence Column!
# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 3, activation = 'sigmoid', input_dim = 8))
# Adding the second hidden layer
classifier.add(Dense(output_dim = 3, activation = 'sigmoid'))
# Adding the output layer
classifier.add(Dense(output_dim = 10, activation = 'sigmoid'))


#print_weights = LambdaCallback(on_epoch_end=lambda batch, logs: print(classifier.layers[2].get_weights()))

# Compiling the ANN
classifier.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100, verbose=1, callbacks = [weightCall])

plt.title("Weights per iteration of last layer")
plt.xlabel("Epochs")
plt.ylabel("Weights")
plt.plot(bias, label = "bias")
plt.plot(w1, label = "w1")
plt.plot(w2, label = "w2")
plt.plot(w3, label = "w3")
plt.legend()
plt.show()
#print(X_train.shape)

#print("FINAL WEIGHTS")
#for layer in classifier.layers:
#    weights = layer.get_weights()
#print(weights)
