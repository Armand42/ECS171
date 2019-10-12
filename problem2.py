#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 10:29:33 2019

@author: armandnasserischool
"""
import numpy as np
import pandas as pd
import seaborn as sns

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

# Assign each column as a set of features
mpg = dataset.iloc[:,0].values
cyl = dataset.iloc[:,1].values
disp = dataset.iloc[:,2].values
hp = dataset.iloc[:,3].values
wt = dataset.iloc[:,4].values
acc = dataset.iloc[:,5].values
yr = dataset.iloc[:,6].values
org = dataset.iloc[:,7].values

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

# plotting data
sns.pairplot(dataset.loc[:,dataset.columns != 'mpg'], hue='threshold')

