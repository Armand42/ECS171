#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 24 20:25:24 2019

@author: armandnasserischool
"""

import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
import warnings
warnings.filterwarnings("ignore")

# Import the dataset and add labels
dataset = pd.read_csv('yeast.data', delim_whitespace=True, 
                       names=["MCG","GVH","ALM","MIT","ERL","POX","VAC","NUC","Class Distribution"])
# X features

X = dataset.drop(columns = 'Class Distribution')

# One Class SVM
def svmOutlier(X):
    y = dataset['Class Distribution']
    clf = OneClassSVM(random_state = 0).fit(X)
    clf.fit(X)
    # Removing all the outliers
    # Prints the new dataset of removed outliers 
    index = 0
    num_out = 0
    for i in clf.predict(X):
        if (i < 0 and index < len(X)):
            num_out += 1
            X = X.drop(X.index[index])
            y = y.drop(y.index[index])
        index = index + 1
    print("The percentage of outliers for SVM detection is: ",num_out/1484 * 100,"%")
    return num_out


# Isolation Forest
def isolOutlier(X):
    y = dataset['Class Distribution']
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
    print("The percentage of outliers for Isolation Forest Detection are: ",num_out/1484 * 100,"%")
  
    return num_out

# Print the percentange of outliers
svmOutlier(X)
isolOutlier(X)




