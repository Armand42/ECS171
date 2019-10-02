#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 23:15:55 2019

@author: armandnasserischool
"""
# 1) Sort column, divide by 4 to find 4 threshold values find cutoff point after each division
# 2) 49 scatter plots in total 7*7 for features not mpg
# save information from low, med, high and create columns for each
# scatter matrix or seaborn
# one libary generates all 7 by 7, permutes all 
# 3) Use the formula
# 4) what 4 lines
# OLS will always give you the best fit
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
# drop Na to specify rows with question marks!!
dataset = pd.read_csv('auto-mpg.data', delim_whitespace=True)

mpg = dataset.iloc[:,0].values
cyl = dataset.iloc[:,1].values
disp = dataset.iloc[:,2].values
hp = dataset.iloc[:,3].values
wt = dataset.iloc[:,4].values
acc = dataset.iloc[:,5].values
yr = dataset.iloc[:,6].values
org= dataset.iloc[:,7].values

# Problem 1 
# Load all MPG values and sort to find cutoff threshold
sortedMPG = np.sort(mpg)
low = sortedMPG[97]
med = sortedMPG[194]
high = sortedMPG[291]
vhigh = sortedMPG[391]
print("low =",low,"med =",med,"high =",high, "very high =",vhigh)

# Assigning new column based on threshold value for Problem 2 graphs
# Needs to adjust the boundary conditions
# low
dataset.loc[mpg <= low, 'threshold'] = "low"

dataset.loc[np.logical_and(mpg > low, mpg < high), 'threshold']= "med"

dataset.loc[np.logical_and(mpg > med, mpg < vhigh), 'threshold']= "high"
# very high
dataset.loc[mpg > high, 'threshold'] = "very high" 

#for i in np.nditer(mpg):
#    if (i <= 17):
#        dataset['thresholds'] = low
#    dataset['thresholds']= 0

# Problem 2
# Need to generate 49 plots

#for pos, axis1 in enumerate(cyl):   # Pick a first col
#    for axis2 in enumerate(disp[pos+1:]):   # Pick a later col
#        plt.scatter(dataset.iloc[:, axis1], dataset.iloc[:, axis2])

   




# sample data weight vs mpg
#x = dataset.iloc[:, 4].values
#y = dataset.iloc[:, 0].values
#x = np.array([0,1,2,3,4,5,6,7,8,9])
#y = np.array([1,2,3,5,7,8,9,10,11,12])

# Least Squares Formula
# size coefficient
#k = np.size(x)

# average of the data
#xbar = np.mean(x)
#ybar = np.mean(y)

# Formula
#numerator = np.sum((x*y)) - (k*xbar*ybar)
#denominator= np.sum((x*x)) - k*(xbar*xbar) 

# Change B to w
#B = numerator/denominator
#B0 = ybar - B*xbar
#print(B)
#print(B0)

    
# Visualising the Test set results
#plt.scatter(x, y, color = 'red')
# Predicted value is equation we generated
#plt.plot(x, B0 + B*x, color = 'blue')
#plt.title('X vs Y (Training set)')
#plt.xlabel('X')
#plt.ylabel('Y')
#plt.show()
       

# #1) 
# mpg | cylinder | displacement | horspower | weight | acceleration | year | origin
# 4 equal sized samples, need to find bounds