#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 11:12:33 2019

@author: armandnasserischool
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import inv
from sklearn.metrics import mean_squared_error
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

# Shuffle dataset before splitting
mpg = shuffle(dataset.iloc[:,0].values, random_state = 0)
cyl = shuffle(dataset.iloc[:,1].values, random_state = 0)
disp = shuffle(dataset.iloc[:,2].values, random_state = 0)
hp = shuffle(dataset.iloc[:,3].values, random_state = 0)
wt = shuffle(dataset.iloc[:,4].values,random_state = 0)
acc = shuffle(dataset.iloc[:,5].values, random_state = 0)
yr = shuffle(dataset.iloc[:,6].values, random_state = 0)
org= shuffle(dataset.iloc[:,7].values, random_state = 0)

# Load all MPG values and sort to find cutoff threshold
sortedMPG = np.sort(mpg)
low = sortedMPG[97]
med = sortedMPG[194]
high = sortedMPG[291]
vhigh = sortedMPG[391] 

# low
dataset.loc[mpg <= low, 'threshold'] = "low"
# med
dataset.loc[np.logical_and(mpg > low, mpg <= med), 'threshold']= "med"
# high
dataset.loc[np.logical_and(mpg > med, mpg < high), 'threshold']= "high"
# very high
dataset.loc[mpg >= high, 'threshold'] = "very high" 

# Inserting a column of ones called bias
dataset.insert(10, 'bias', 1)
bias = dataset.iloc[:,10].values

# Splitting up training and test data for each independent variable
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

# This function will simulataneously compute 0,1,2,3 degrees for a single independent variable
def linReg(x,y):
   
    # degree 0 weights are simply the mean
    degreeZero = mpg.mean()
    plt.axhline(y=degreeZero, color='y')

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
    

    # degree 2
    # Applying the OLS Formula to acquire 2nd degree weights
    n22 =np.column_stack((n11, x**2))
    n2Transposed = np.transpose(n22)
    n22Term1 = np.dot(n2Transposed,n22)
    invertedn2 = inv(n22Term1)
    n22Term2 = np.dot(n2Transposed,y)
    degreeTwo = np.dot(invertedn2,n22Term2)
    
    # degree 3 
    # Applying the OLS Formula to acquire 3rd degree weights
    n222 = np.column_stack((n111, x**2))
    n333 = np.column_stack((n222, x**3))
    n3Transposed = np.transpose(n333)
    n33Term1 = np.dot(n3Transposed,n333)
    invertedn3 = inv(n33Term1)
    n33Term2 = np.dot(n3Transposed,y)
    degreeThree = np.dot(invertedn3,n33Term2)
   
    
    # Create a scatter plot for all degrees
    plt.scatter(x, y, color = "black", marker = "*", s = 30)
    
    return degreeZero, degreeOne, degreeTwo, degreeThree

# This function will simultaneously display all the training graphs for the training data
def plotTrainData(xtrain, ytrain):
    
    # Takes in the return values of the training data
    data = linReg(xtrain, ytrain)
    # Extracting weights from all degrees
    degreeZero = data[0]
    degreeOne = data[1]
    degreeTwo = data[2]
    degreeThree = data[3]
   
    w0 = degreeOne[0]
    w1 = degreeOne[1]
    
    w00 = degreeTwo[0]
    w11 = degreeTwo[1]
    w22 = degreeTwo[2]
    
    w000 = degreeThree[0]
    w111 = degreeThree[1]
    w222 = degreeThree[2]
    w333 = degreeThree[3]
    # Calculating the mean squared error for each set of features
    print("Oth Order MSE Yellow (Training):", mean_squared_error(ytrain,degreeZero*biasTrain))
    print("1st Order MSE Red (Training):",mean_squared_error(ytrain,w0+xtrain*w1)) 
    print("2nd Order MSE Blue (Training):",mean_squared_error(ytrain,w00+xtrain*w11 +w22*xtrain**2)) 
    print("3rd Order MSE Green (Training):",mean_squared_error(ytrain,w000+xtrain*w111+ w222*xtrain**2 + w333*xtrain**3)) 
    
    # Sort data for ensuring correct regression displays
    xtrain = np.sort(xtrain)
    ytrain = np.sort(ytrain)
    
    # Plot all the training graphs
    plt.plot(xtrain,w0+xtrain*w1, color = "red")
    plt.plot(xtrain,w00+xtrain*w11 +w22*xtrain**2, color = "blue")
    plt.plot(xtrain,w000+xtrain*w111+ w222*xtrain**2 + w333*xtrain**3, color = "green")
    #plt.title('(Training set)')
    #plt.xlabel('input')
    plt.ylabel('mpg')
    
    plt.show()
    

# This function will simultaneously display all the training graphs for the testing data
def plotTestData(xtest,ytest):
    # Extracting weights from all degrees
    data = linReg(xtest, ytest)
    
    degreeZero = data[0]
    degreeOne = data[1]
    degreeTwo = data[2]
    degreeThree = data[3]
    
    w0 = degreeOne[0]
    w1 = degreeOne[1]
    
    w00 = degreeTwo[0]
    w11 = degreeTwo[1]
    w22 = degreeTwo[2]
    
    w000 = degreeThree[0]
    w111 = degreeThree[1]
    w222 = degreeThree[2]
    w333 = degreeThree[3]
    
    plt.scatter(xtest, ytest, color = "black", marker = "*", s = 30)
    # Calculating the mean squared error for each set of features
    print("Oth Order MSE Yellow (Testing):", mean_squared_error(ytest,degreeZero*biasTest))
    print("1st Order MSE Red (Testing):",mean_squared_error(ytest,w0+xtest*w1)) 
    print("2nd Order MSE Blue (Testing):",mean_squared_error(ytest,w00+xtest*w11 +w22*xtest**2)) 
    print("3rd Order MSE Green (Testing):",mean_squared_error(ytest,w000+xtest*w111+ w222*xtest**2 + w333*xtest**3))
   
    # Sort for ensuring correct regression displays
    xtest = np.sort(xtest)
    ytest = np.sort(ytest)
    
    # Plot all the testing graphs
    
    plt.axhline(y=degreeZero, color='y')
    plt.plot(xtest,w0+xtest*w1, color = "red")
    plt.plot(xtest,w00+xtest*w11 +w22*xtest**2, color = "blue")
    plt.plot(xtest,w000+xtest*w111+ w222*xtest**2 + w333*xtest**3, color = "green")
    #plt.title('(Testing set)')
    #plt.xlabel('input variable')
    plt.ylabel('mpg')
    plt.show()
    
# Printing and displaying all the test/train graphs and the test/train MSE's
    
print("Cylinder MSE:")
plt.title('MPG vs CYL (Training Set)')
plt.xlabel('cylinder')
trainReg1 = plotTrainData(cyl_train,mpg_train)
print("Cylinder MSE:")
plt.title('MPG vs CYL (Testing Set)')
plt.xlabel('cylinder')
testReg1 = plotTestData(cyl_test, mpg_test)

print("Displacement MSE:")
plt.title('MPG vs DISP (Training Set)')
plt.xlabel('displacement')
trainReg2 = plotTrainData(disp_train,mpg_train)
print("Displacement MSE:")
plt.title('MPG vs DISP  (Testing Set)')
plt.xlabel('displacement')
testReg2 = plotTestData(disp_test, mpg_test)


print("Horsepower MSE:")
plt.title('MPG vs HP (Training Set)')
plt.xlabel('horsepower')
trainReg3 = plotTrainData(hp_train,mpg_train)
print("Horsepower MSE:")
plt.title('MPG vs HP (Testing Set)')
plt.xlabel('horsepower')
testReg3 = plotTestData(hp_test, mpg_test)


print("Weight MSE:")
plt.title('MPG vs WT (Training Set)')
plt.xlabel('weight')
trainReg4 = plotTrainData(wt_train,mpg_train)
print("Weight MSE:")
plt.title('MPG vs WT (Testing Set)')
plt.xlabel('weight')
testReg4 = plotTestData(wt_test, mpg_test)


print("Acceleration MSE:")
plt.title('MPG vs ACC (Training Set)')
plt.xlabel('acceleration')
trainReg5 = plotTrainData(acc_train,mpg_train)
print("Acceleration MSE:")
plt.title('MPG vs ACC (Testing Set)')
plt.xlabel('acceleration')
testReg5 = plotTestData(acc_test, mpg_test)


print("Year MSE:")
plt.title('MPG vs YR (Training Set)')
plt.xlabel('year')
trainReg6 = plotTrainData(yr_train,mpg_train)
print("Year MSE:")
plt.title('MPG vs YR (Testing Set)')
plt.xlabel('year')
testReg6 = plotTestData(yr_test, mpg_test)


print("Origin MSE:")
plt.xlabel('origin')
plt.title('MPG vs ORG (Training Set)')
trainReg7 = plotTrainData(org_train,mpg_train)
print("Origin MSE:")
plt.title('MPG vs ORG (Testing Set)')
plt.xlabel('origin')
testReg7 = plotTestData(org_test, mpg_test)
