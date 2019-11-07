#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  4 17:21:02 2019

@author: armandnasserischool
"""

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




# Dynamically create a classifier
def createClassifier(X_train, y_train, inp, nodes, epochs, batch, outLayer, numHidden):
    # Initialising the ANN
    classifier = Sequential()
    # First Hidden Layer
    classifier.add(Dense(output_dim = nodes, activation = 'relu', input_dim = inp))
    for dense in range(numHidden - 1):
        classifier.add(Dense(output_dim = nodes, activation = 'relu', input_dim = inp))
    # Adding the output layer
    classifier.add(Dense(output_dim = outLayer, activation = 'softmax'))
    # Compiling the ANN
    classifier.compile(optimizer = 'sgd', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # Return a classifier object
    return classifier

# Function to iteratively perform grid search and return the testing error values
def performGridSearch(classifier):
    errorValues = []
    combination = []
    # Layers and Nodes
    numHidden = [1,2,3]
    nodes = [3,6,9,12]
    for hid in numHidden:
        for n in nodes:
            # Fit the classifier for each combination and calculate the testing error simulataneously
            model = createClassifier(X_train,y_train,8,n,100,10,10,hid)
            history = model.fit(X_train, y_train, batch_size = 10, nb_epoch = 100, verbose=1, callbacks = [weightCall], validation_data = (X_test,y_test))
            test_loss = [1-x for x in history.history['val_acc']]
            train_loss = [1-x for x in history.history['acc']]
            temp = 1 - model.evaluate(X_train,y_train)[1]
            
            # Append testing errors and combinations to arrays
            testingError = temp
            errorValues.append(testingError)
            combination.append((hid,n))
            # Print out the Hidden Layer & Node Combination & the Testing Error
            print("The testing error for",hid,"hidden layers and",n,"nodes is:",testingError)
            print("Hidden value is:", hid)
            print("Node value is :", n)
            
            
           
    #Plotting the Weights per iteration of the last layer
    plt.title("Training & Testing Error per Iteration")
    plt.plot(train_loss, label = "Train Loss", color = "green")
    plt.plot(test_loss, label = "Test Loss", color = "blue")
    plt.xlabel("Epochs")
    plt.ylabel("Ratio")
    plt.legend()
    plt.show()
          
    return errorValues

# Instantiating a classifier with sample data
testModel = createClassifier(X_train, y_train,8,3,100,10,10,2)

result = performGridSearch(testModel)

