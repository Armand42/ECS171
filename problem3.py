#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:50:17 2019

@author: armandnasserischool
"""

import numpy as np
import pandas as pd
from numpy.linalg import inv
from sklearn.utils import shuffle

# Import the dataset and add labels
dataset = pd.read_csv('auto-mpg.data', delim_whitespace=True, 
                       names=["mpg","Cylinder","Displacement","Horsepower","Weight","Acceleration",
                              "Year","Origin","Model"])
# Remove all ?'s
dataset = dataset.drop(dataset[dataset['Horsepower'] == '?'].index)
# Resest indexes
dataset = dataset.reset_index(drop=True)
# Typecast Horsepower column back to float
dataset['Horsepower'] = dataset['Horsepower'].apply(pd.to_numeric, errors='coerce')


# Assign each column a set of features and shuffling the dataset
mpg = shuffle(dataset.iloc[:,0].values, random_state = 0)
cyl = shuffle(dataset.iloc[:,1].values, random_state = 0)
disp = shuffle(dataset.iloc[:,2].values, random_state = 0)
hp = shuffle(dataset.iloc[:,3].values, random_state = 0)
wt = shuffle(dataset.iloc[:,4].values,random_state = 0)
acc = shuffle(dataset.iloc[:,5].values, random_state = 0)
yr = shuffle(dataset.iloc[:,6].values, random_state = 0)
org= shuffle(dataset.iloc[:,7].values, random_state = 0)

# Sort and find the cutoff threshold for each category
sortedMPG = np.sort(mpg)
low = sortedMPG[97]
med = sortedMPG[194]
high = sortedMPG[291]
vhigh = sortedMPG[391] 

# Assigning each category to mpg values
# low
dataset.loc[mpg <= low, 'threshold'] = "low"
# med
dataset.loc[np.logical_and(mpg > low, mpg <= med), 'threshold']= "med"
# high
dataset.loc[np.logical_and(mpg > med, mpg < high), 'threshold']= "high"
# very high
dataset.loc[mpg >= high, 'threshold'] = "very high" 

# Create a column of ones for bias
dataset.insert(10, 'bias', 1)
bias = dataset.iloc[:,10].values

# Splitting up train and test data
biasTrain = bias[0:292]
biasTest = bias[292:392]

mpg_train = mpg[0:292]
mpg_test = mpg[292:392]

# mpg vs. cyl
cyl_train = cyl[0:292]
cyl_test = cyl[292:392]

# mpg vs. disp
disp_train = disp[0:292]
disp_test = disp[292:392]

# mpg vs. hp
hp_train = hp[0:292]
hp_test = hp[292:392]

# mpg vs. wt
wt_train = wt[0:292]
wt_test = wt[292:392]

# mpg vs. acc
acc_train = acc[0:292]
acc_test = acc[292:392]

# mpg vs. yr
yr_train = yr[0:292]
yr_test = yr[292:392]

# mpr vs. org
org_train = org[0:292]
org_test = org[292:392]


def linReg(x,y):
   
    # degree 0 weights are simply the mean
    degreeZero = mpg.mean()
    
    # degree 1
    # Checking sizes for bias for train or test for to resolve dimension issues with train/test data
    if (x.size < 101 and y.size < 101):
        n1 = np.column_stack((biasTest, x))
        n11 = np.column_stack((biasTest, x))
        n111 = np.column_stack((biasTest, x))
    else:
        n1 = np.column_stack((biasTrain, x))
        n11 = np.column_stack((biasTrain, x))
        n111 = np.column_stack((biasTrain, x))
    
    # Applying the OLS Formula to acquire 1st degree weights
    n1Transposed = np.transpose(n1)
    n1Term1 = np.dot(n1Transposed,n1)
    invertedn1 = inv(n1Term1)
    n2Term2 = np.dot(n1Transposed,y)
    degreeOne = np.dot(invertedn1,n2Term2)
    w0 = degreeOne[0]
    w1 = degreeOne[1]

    # degree 2
    # Applying the OLS Formula to acquire 2nd degree weights
    n22 =np.column_stack((n11, x**2))
    n2Transposed = np.transpose(n22)
    n22Term1 = np.dot(n2Transposed,n22)
    invertedn2 = inv(n22Term1)
    n22Term2 = np.dot(n2Transposed,y)
    degreeTwo = np.dot(invertedn2,n22Term2)
    w00 = degreeTwo[0]
    w11 = degreeTwo[1]
    w22 = degreeTwo[2]

    # degree 3 
    # Applying the OLS Formula to acquire 3rd degree weights
    n222 = np.column_stack((n111, x**2))
    n333 = np.column_stack((n222, x**3))
    n3Transposed = np.transpose(n333)
    n33Term1 = np.dot(n3Transposed,n333)
    invertedn3 = inv(n33Term1)
    n33Term2 = np.dot(n3Transposed,y)
    degreeThree = np.dot(invertedn3,n33Term2)
    w000 = degreeThree[0]
    w111 = degreeThree[1]
    w222 = degreeThree[2]
    w333 = degreeThree[3]
    
    # Printing and returning weights
    print("Oth Order Weights:",degreeZero)
    print("1st Order Weights:",w0,w1) 
    print("2nd Order Weights:",w00,w11,w22) 
    print("3rd Order Weights:",w000,w111,w222,w333) 
    
  
    return degreeZero, degreeOne, degreeTwo, degreeThree

# Testing the linear regression function with sample data
testRegression = linReg(wt_train,mpg_train)