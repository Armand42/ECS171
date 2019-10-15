#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 19:16:02 2019

@author: armandnasserischool
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.utils import shuffle
from sklearn.metrics import precision_score
from sklearn.preprocessing import MinMaxScaler

# Import the dataset and add labels
dataset = pd.read_csv('auto-mpg.data', delim_whitespace=True, 
                       names=["mpg","Cylinder","Displacement","Horsepower","Weight","Acceleration",
                              "Year","Origin","Model"])
# Remove all ?'s
dataset = dataset.drop(dataset[dataset['Horsepower'] == '?'].index)
# Reset indexes
dataset = dataset.reset_index(drop=True)

# Typecast Horsepower column back to float
dataset['Horsepower'] = dataset['Horsepower'].apply(pd.to_numeric, errors='coerce')

# Assign each column as a set of features
mpg = dataset.iloc[:,0].values
cyl = dataset.iloc[:,1].values
disp = dataset.iloc[:,2].values
hp = dataset.iloc[:,3].values
wt = dataset.iloc[:,4].values
acc = dataset.iloc[:,5].values
yr = dataset.iloc[:,6].values
org= dataset.iloc[:,7].values

# Load all MPG values and sort to find cutoff threshold
sortedMPG = np.sort(mpg)
low = sortedMPG[97]
med = sortedMPG[194]
high = sortedMPG[291]
vhigh = sortedMPG[391] 

# low
dataset.loc[mpg <= low, 'threshold'] = 0
# med
dataset.loc[np.logical_and(mpg > low, mpg <= med), 'threshold']= 1
# high
dataset.loc[np.logical_and(mpg > med, mpg < high), 'threshold']= 2
# very high
dataset.loc[mpg >= high, 'threshold'] = 3 

# Shuffling the dataset
dataset = shuffle(dataset,random_state = 0)
# Splitting up the data
xFeatures = dataset.iloc[:,1:8].values

# Applying normalization to the dataset
scaler = MinMaxScaler()
scaler.fit(xFeatures)

# Saving scaled dataset
scaledData = scaler.transform(xFeatures)

# Saving mpg categories
yLabel = dataset['threshold']

# Splitting up the test and train data with normalized dataset
X_train = scaledData[0:292]
y_train = yLabel[0:292]

X_test = scaledData[292:392]
y_test = yLabel[292:392]

# Applying the Logistic Regressor to the training and testing data
logregTrain = LogisticRegression()
#logregTest = LogisticRegression()

# Fitting the regression onto the appropriate data
train = logregTrain.fit(X_train,y_train)
#test = logregTest.fit(X_test,y_test)

# Calculating the predicted values
y_predTrain=logregTrain.predict(X_train)
y_pred=logregTrain.predict(X_test)

# Printing out the calculated precision values
print("Precision (Training):",precision_score(y_train, y_predTrain, average = 'micro'))
print("Precision (Testing):",precision_score(y_test, y_pred, average = 'micro'))


