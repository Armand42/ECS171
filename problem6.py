#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 12 14:36:02 2019

@author: armandnasserischool
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score

# Import the dataset and add labels
dataset = pd.read_csv('auto-mpg.data', delim_whitespace=True, 
                       names=["mpg","Cylinder","Displacement","Horsepower","Weight","Acceleration",
                              "Year","Origin","Model"])
# Remove all ?'s
dataset = dataset.drop(dataset[dataset['Horsepower'] == '?'].index)
# Rest indexes
dataset = dataset.reset_index(drop=True)
#option 2 but this will make Model column Nan
#dataset=dataset.apply(pd.to_numeric, errors='coerce')
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

# Splitting up the data
xFeatures = dataset.iloc[:,1:8].values 
yLabel = dataset['threshold']
# Splitting up the test and train data
X_train,X_test,y_train,y_test=train_test_split(xFeatures,yLabel,test_size=0.2552,random_state=0)

# Applying the Logistic Regressor to the training data
logreg = LogisticRegression(multi_class='ovr')

train = logreg.fit(X_train,y_train)
test = logreg.fit(X_test,y_test)
# Calculating the predicted values
y_predTrain=logreg.predict(X_train)
y_pred=logreg.predict(X_test)

# Printing out the calculated precision values
print("Overall Precision (Training):",precision_score(y_train, y_predTrain, average = 'micro'))
print("Separate Bin Precision (Training):",precision_score(y_train, y_predTrain, average = None))
print("Overall Precision (Testing):",precision_score(y_test, y_pred, average = 'micro'))
print("Separate Bin Precision (Testing):",precision_score(y_test, y_pred, average = None))
