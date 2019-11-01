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
from keras.models import Sequential
from keras.layers import Dense
from sklearn.utils import shuffle

# NEED ONE HOT ENCODER
# Am I doing this right?
# How to generate charts exactly?
# Import the dataset and add labels
dataset = pd.read_csv('yeast.data', delim_whitespace=True, 
                       names=["Sequence Name","MCG","GVH","ALM","MIT","ERL","POX","VAC","NUC","Class Distribution"])

# X features and y categories
X = dataset.iloc[:,1:9].values
y = dataset.iloc[:,9]
# Encoding categorical data, need 1 hot encoder
labelEncoder = preprocessing.LabelEncoder()
y = labelEncoder.fit_transform(y)



# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.34, random_state = 0)

# Initialising the ANN
classifier = Sequential()
# Drop Sequence Column!
# Adding the input layer and the first hidden layer
classifier.add(Dense(activation = 'sigmoid', input_dim = 8))

# Adding the second hidden layer
classifier.add(Dense(input_dim = 3, activation = 'sigmoid'))
# Adding the second hidden layer
classifier.add(Dense(input_dim = 3, activation = 'sigmoid'))

# Adding the output layer
classifier.add(Dense(input_dim = 10, activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 32, nb_epoch = 5000)

