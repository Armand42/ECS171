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
clf = OneClassSVM(gamma='auto').fit(X)
clf.predict(X)
clf.fit(X)
#print(X)
# Isolation Forest
isol = IsolationForest(n_estimators=8, warm_start=True)
isol.fit(X)
# Removing all the outliers
# Prints the new dataset of removed outliers 
index = 0
for i in isol.predict(X):
    if (i < 0 and index < len(X)):
        X = X.drop(X.index[index])
    index = index + 1
print(X)


