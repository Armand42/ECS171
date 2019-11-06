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
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import keras
from matplotlib import pyplot as plt
from keras.models import Sequential
from keras.wrappers.scikit_learn import KerasClassifier
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


y = pd.get_dummies(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.34, random_state = 0)


# Dynamically create a classifier
def createClassifier(X_train, y_train, inp=8, nodes=3, epochs=100, batch=10, outLayer=10, numHidden=2):
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

    history = classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100, verbose=1, callbacks = [weightCall], validation_data = (X_test,y_test))
    test_loss = [1-x for x in history.history['val_acc']]
    train_loss = [1-x for x in history.history['acc']]

    plt.title("Training & Testing Error per Iteration")
    plt.plot(train_loss, label = "Train Loss", color = "green")
    plt.plot(test_loss, label = "Test Loss", color = "blue")
    plt.xlabel("Epochs")
    plt.ylabel("Ratio")
    plt.legend()
    plt.show()
    return history 

createClassifier(X_train, y_train,8,3,100,10,10,2)

