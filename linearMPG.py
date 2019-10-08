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
sns.pairplot(dataset.loc[:,dataset.columns != 'mpg'], hue='threshold')
plt.legend(loc='lower right')



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
#constant = biasTrain
#constantTranspose = np.transpose(constant)
#constantTerm1 = np.dot(constantTranspose,constant)
#invertedConstant = inv(constantTerm1)
#constantTerm2 = np.dot(constantTranspose,mpg_train)
#constantResult = np.dot(invertedConstant,constantTerm2)

# Comment out sns pairplot option 2 to see individual graphs
# Need to create test bias now
# MSE
# Rename variables pleassssse
def linReg(x,y):
    

# degree 1
    n1 = np.column_stack((biasTrain, x))
    n1Transposed = np.transpose(n1)
    n1Term1 = np.dot(n1Transposed,n1)
    invertedn1 = inv(n1Term1)
    n2Term2 = np.dot(n1Transposed,y)
    degreeOne = np.dot(invertedn1,n2Term2)
    w0 = degreeOne[0]
    w1 = degreeOne[1]

# degree 2
    n11 = np.column_stack((biasTrain, x))
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
    n111 = np.column_stack((biasTrain, x))
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
    
    plt.scatter(x, y, color = "m", marker = "o", s = 30)
    plt.plot(x,w0+x*w1, color = "red")
    plt.plot(x,w00+x*w11 +w22*x**2, color = "blue")
    plt.plot(x,w000+x*w111+ w222*x**2 + w333*x**3, color = "green")
    plt.title('(Training set)')
    plt.xlabel('input variable')
    plt.ylabel('mpg')
    plt.show()


# Still need degree 0 line??
testReg1 = linReg(cyl_train, mpg_train)
testReg2 = linReg(disp_train, mpg_train)
testReg3 = linReg(hp_train, mpg_train)
testReg4 = linReg(wt_train, mpg_train)
testReg5 = linReg(acc_train, mpg_train)
testReg6 = linReg(yr_train, mpg_train)
testReg7 = linReg(org_train, mpg_train)


