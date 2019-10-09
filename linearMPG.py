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
from sklearn.metrics import mean_squared_error
from sklearn.utils import shuffle
# drop Na to specify rows with question marks!!
# figure out how to add row of names
# Import the dataset
dataset = pd.read_csv('auto-mpg.data', delim_whitespace=True)
# Store each column into separate independent variables



mpg = shuffle(dataset.iloc[:,0].values, random_state = 0)
cyl = shuffle(dataset.iloc[:,1].values, random_state = 0)
disp = shuffle(dataset.iloc[:,2].values, random_state = 0)
hp = shuffle(dataset.iloc[:,3].values, random_state = 0)
wt = shuffle(dataset.iloc[:,4].values,random_state = 0)
acc = shuffle(dataset.iloc[:,5].values, random_state = 0)
yr = shuffle(dataset.iloc[:,6].values, random_state = 0)
org= shuffle(dataset.iloc[:,7].values, random_state = 0)



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
# DONT FORGET THISSSSSSSSS
# Option 2

#sns.pairplot(dataset.loc[:,dataset.columns != 'mpg'], hue='threshold')
#plt.legend(loc='lower right')



# Problem 3 Prototype Regression

# Inserting a column of ones called bias
dataset.insert(10, 'bias', 1)
bias = dataset.iloc[:,10].values

# Only used to split bias for appropriate dimensions
#bias_train, bias_test, mpg_train, mpg_test = train_test_split(bias, mpg, test_size = 0.2551)
#biasTrain = bias[0:292]
#biasTest = bias[292:392]


def linearReg(x,y):
    # For ensuring correct dimension size for bias
    if (x.size < 101 and y.size < 101):
        combinedX = np.column_stack((biasTest, x))
    else:
        combinedX = np.column_stack((biasTrain, x))
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
    
    print("1sssst Order MSE:",mean_squared_error(y,w0+x*w1)) 
    plt.show()
  
    
    return w0,w1


# Problem 4
# Splitting up training and test data for each independent variable
# mpg train-test data
    
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





# deg 0
constant = biasTrain
#constantTranspose = np.transpose(constant)
#constantTerm1 = np.dot(constant,constantTranspose)
#invertedConstant = inv(constantTerm1)
#constantTerm2 = np.dot(constantTranspose,mpg_train)
#constantResult = np.dot(invertedConstant,constantTerm2)

# COMMENT OUT SNS PAIRPLOT option 2 to see individual graphs
# Need to create test bias now
# MSE
# Rename variables pleassssse

def linReg(x,y):
   
# degree 0
    degreeZero = mpg.mean()
    plt.axhline(y=degreeZero, color='y')

# degree 1
    # Checking sizes for bias for train or test
    if (x.size < 101 and y.size < 101):
        n1 = np.column_stack((biasTest, x))
        n11 = np.column_stack((biasTest, x))
        n111 = np.column_stack((biasTest, x))
    else:
        n1 = np.column_stack((biasTrain, x))
        n11 = np.column_stack((biasTrain, x))
        n111 = np.column_stack((biasTrain, x))
    
    #n1 = np.column_stack((biasTrain, x))
    n1Transposed = np.transpose(n1)
    n1Term1 = np.dot(n1Transposed,n1)
    invertedn1 = inv(n1Term1)
    n2Term2 = np.dot(n1Transposed,y)
    degreeOne = np.dot(invertedn1,n2Term2)
    w0 = degreeOne[0]
    w1 = degreeOne[1]

# degree 2
    #n11 = np.column_stack((biasTrain, x))
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
    #n111 = np.column_stack((biasTrain, x))
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
    
    plt.scatter(x, y, color = "black", marker = "*", s = 30)
    
    #print(mean_squared_error(degreeZero,degreeZero)) 
    print("Oth Order MSE: ???")
    print("1st Order MSE (Training):",mean_squared_error(y,w0+x*w1)) 
    print("2nd Order MSE (Training):",mean_squared_error(y,w00+x*w11 +w22*x**2)) 
    print("3rd Order MSE (Training):",mean_squared_error(y,w000+x*w111+ w222*x**2 + w333*x**3)) 
    
        

    plt.plot(x,w0+x*w1, color = "red")
    plt.plot(x,w00+x*w11 +w22*x**2, color = "blue")
    plt.plot(x,w000+x*w111+ w222*x**2 + w333*x**3, color = "green")
    plt.title('(Training set)')
    plt.xlabel('input variable')
    plt.ylabel('mpg')
    plt.show()
#    
    return degreeZero, degreeOne, degreeTwo, degreeThree

def plotTestData(xtest,ytest):
    # Takes in the return values of the training data
    data = linReg(xtest, ytest)
    
    degreeZero = data[0]
    degreeOne = data[1]
    degreeTwo = data[2]
    degreeThree = data[3]
    # Assigning weights to test data
    w0 = degreeOne[0]
    w1 = degreeOne[1]
    
    w00 = degreeTwo[0]
    w11 = degreeTwo[1]
    w22 = degreeTwo[2]
    
    w000 = degreeThree[0]
    w111 = degreeThree[1]
    w222 = degreeThree[2]
    w333 = degreeThree[3]
    
    plt.scatter(xtest, ytest, color = "m", marker = "*", s = 30)
    print("1st Order MSE (Testing):",mean_squared_error(ytest,w0+xtest*w1)) 
    print("2nd Order MSE (Testing):",mean_squared_error(ytest,w00+xtest*w11 +w22*xtest**2)) 
    print("3rd Order MSE (Testing):",mean_squared_error(ytest,w000+xtest*w111+ w222*xtest**2 + w333*xtest**3))
   
    plt.axhline(y=degreeZero, color='y')
    plt.plot(xtest,w0+xtest*w1, color = "red")
    plt.plot(xtest,w00+xtest*w11 +w22*xtest**2, color = "blue")
    plt.plot(xtest,w000+xtest*w111+ w222*xtest**2 + w333*xtest**3, color = "green")
    plt.title('(Testing set)')
    plt.xlabel('input variable')
    plt.ylabel('mpg')
    plt.show()
    
# Issues:
    # FIX 1st order
    # Are my MSEs correct?
    # How to calculate 0th order MSE
    # How to calculate training error
    
# How to do 5 and 6
# Dont forget that you commented out the big chart and testers!!!


# Training Regressions
print("Cylinder MSE:")
trainReg1 = linReg(cyl_train, mpg_train)
print("Displacement MSE:")
trainReg2 = linReg(disp_train, mpg_train)
print("Horsepower MSE:")
trainReg3 = linReg(hp_train, mpg_train)
print("Weight MSE:")
trainReg4 = linReg(wt_train, mpg_train)
print("Acceleration MSE:")
trainReg5 = linReg(acc_train, mpg_train)
print("Year MSE:")
trainReg6 = linReg(yr_train, mpg_train)
print("Origin MSE:")
trainReg7 = linReg(org_train, mpg_train)


# Testing Regressions
#print("TESTING REGRESSIONS")
#testReg1 = plotTestData(cyl_test, mpg_test)
#testReg2 = plotTestData(disp_test, mpg_test)
#testReg3 = plotTestData(hp_test, mpg_test)
#testReg4 = plotTestData(wt_test, mpg_test)
#testReg5 = plotTestData(acc_test, mpg_test)
#testReg6 = plotTestData(yr_test, mpg_test)
#testReg7 = plotTestData(org_test, mpg_test)

#mean_squared_error(wt_train,wt_test) 
