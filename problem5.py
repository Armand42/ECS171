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
np.random.seed(0)
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
def createClassifier(X_train, y_train, inp, nodes, epochs, batch, outLayer, numHidden):
    # Initialising the ANN
    classifier = Sequential()
    # First Hidden Layer
    classifier.add(Dense(output_dim = nodes, activation = 'sigmoid', input_dim = inp))
    for dense in range(numHidden - 1):
        classifier.add(Dense(output_dim = nodes, activation = 'sigmoid', input_dim = inp))

    # Adding the output layer
    classifier.add(Dense(output_dim = outLayer, activation = 'sigmoid'))
    
    # Compiling the ANN
    classifier.compile(optimizer = 'sgd', loss = 'mean_squared_error', metrics = ['accuracy'])

    #classifier.fit(X_train, y_train, batch_size = batch, nb_epoch = epochs, verbose=1)
    return classifier

def performGridSearch(classifier):
    errorValues = []
    #classifier = createClassifier(X_train, y_train,8,3,100,10,10,2)
    #model = KerasClassifier(build_fn = createClassifier(X_train, y_train,8,3,100,10,10,2))
    numHidden = [1,2,3]
    nodes = [3,6,9,12]
    i = 0
    for hid in numHidden:
        for n in nodes:
            model = createClassifier(X_train,y_train,8,n,100,10,10,hid)
            model.fit(X_train, y_train, batch_size = 10, nb_epoch = 100, verbose=1)
            temp = 1 - model.evaluate(X_train,y_train)[1]
            testingError = temp
            errorValues.append(testingError)
            #errorValues.append(hid)
            #errorValues.append(n)
            print("The testing error for",hid,"hidden layers and",n,"nodes is:",testingError)
            i = i + 1
            print("hidden value is:", hid)
            print("node value is :", n)
            #print(i)
            
    #parameters = dict(numHidden = numHidden, nodes= nodes)
    
    #grid = GridSearchCV(estimator = model, param_grid = parameters, n_jobs=-1, cv = 3)
   
    #grid_result = grid.fit(X_train, y_train)
    #print(grid_result)
    return errorValues



testModel = createClassifier(X_train, y_train,8,4,10,10,10,3)

result = performGridSearch(testModel)

print(np.amax(result))