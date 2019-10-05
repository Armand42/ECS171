#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 23:15:55 2019

@author: armandnasserischool
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from pandas.plotting import scatter_matrix
from numpy.linalg import inv
from sklearn.model_selection import train_test_split
# drop Na to specify rows with question marks!!
# figure out how to add row of names
# Import the dataset
dataset = pd.read_csv('auto-mpg.data', delim_whitespace=True)
# Store each column into separate independent variables
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

# Problem 2
# Assigning new column based on threshold value for Problem 2 Graphs
# Need to generate 49 plots

# low
dataset.loc[mpg <= low, 'threshold'] = "low"
# med
dataset.loc[np.logical_and(mpg > low, mpg <= med), 'threshold']= "med"
# high
dataset.loc[np.logical_and(mpg > med, mpg < high), 'threshold']= "high"
# very high
dataset.loc[mpg >= high, 'threshold'] = "very high" 

#sns.lmplot(x ="Weight",y="mpg", data=dataset, fit_reg=False, hue='threshold', legend=False)
#sns.lmplot(x ="Weight",y="Horsepower", data=dataset, fit_reg=False, hue='threshold', legend=False)
#plt.legend(loc='lower right')
# Option 1
# Actual plot with all variables against each other
#scatter_matrix(dataset.loc[:,'Cylinder':'Origin'], alpha=0.2, figsize=(20, 20))

# Option 2
sns.pairplot(dataset.loc[:, dataset.columns != 'mpg'], hue='threshold')
plt.legend(loc='lower right')



# Problem 3 Prototype Regression

# Inserting a column of ones called bias
dataset.insert(10, 'bias', 1)
bias = dataset.iloc[:,10].values

# Only used to split bias for appropriate dimensions
bias_train, bias_test, mpg_train, mpg_test = train_test_split(bias, mpg, test_size = 0.2551)

def linearReg(x,y):
    # For ensuring correct dimension size for bias
    if (x.size < 101 and y.size < 101):
        combinedX = np.column_stack((bias_test, x))
    else:
        combinedX = np.column_stack((bias_train, x))
    # Transpose x
    Xtransposed = np.transpose(combinedX)
    # First expression ok
    expression1 = np.dot(Xtransposed,combinedX)
    # inverse ok
    inverse = inv(expression1)
    # Second expression ok
    expression2 = np.dot(Xtransposed,y)

    result = np.dot(inverse,expression2)
    w0 = result[0]
    w1 = result[1]
    
    # Eventually need to optimize this
    # Need to modify so that it provides multiple 0,1,2,3
    plt.scatter(x,y, color = "m", marker = "o", s = 30) 
    plt.plot(x, w0+w1*x, color = "green")
    plt.show()
  
    
    return w0,w1

#testReg = linearReg(wt,mpg)
#plt.scatter(wt, mpg, color = "m", marker = "o", s = 30) 
#plt.plot(wt, testReg[0] + testReg[1]*wt, color = "green")
#plt.show()
#print(testReg)


# Problem 4
# Splitting up training and test data for each independent variable
# Not really using numerical mpg train and test, simply placeholder variables

# Cylinder vs. mpg
#cyl_train, cyl_test, mpg_train1, mpg_test1 = train_test_split(cyl, mpg, test_size = 0.2551)
# Displacement vs. mpg
#disp_train, disp_test, mpg_train, mpg_test = train_test_split(disp, mpg, test_size = 0.2551)
# Horsepower vs. mpg
hp_train, hp_test, mpg_train, mpg_test = train_test_split(hp, mpg, test_size = 0.2551)
# Weight vs. mpg
wt_train, wt_test, mpg_train, mpg_test = train_test_split(wt, mpg, test_size = 0.2551)
# Acceleration vs. mpg
#acc_train, acc_test, mpg_train5, mpg_test5 = train_test_split(acc, mpg, test_size = 0.2551)
# Year vs. mpg
#yr_train, yr_test, mpg_train6, mpg_test6= train_test_split(yr, mpg, test_size = 0.2551)
# Origin vs. mpg
#org_train, org_test, mpg_train7, mpg_test7= train_test_split(org, mpg, test_size = 0.2551)

#################################
trainReg = linearReg(wt_train, mpg_train)
plt.scatter(wt_train, mpg_train, color = "m", marker = "o", s = 30) 

plt.plot(wt_train, trainReg[0] + trainReg[1]*wt_train, color = "green")
plt.title('mpg vs. weight (Training set)')
plt.xlabel('weight')
plt.ylabel('mpg')
plt.show()

testReg = linearReg(wt_test, mpg_test)
plt.scatter(wt_test, mpg_test, color = "m", marker = "o", s = 30) 

plt.plot(wt_test, testReg[0] + testReg[1]*wt_test, color = "green")
plt.title('mpg vs. weight (Test set)')
plt.xlabel('weight')
plt.ylabel('mpg')
plt.show()
################################3






    
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