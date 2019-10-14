#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 12:08:37 2019

@author: armandnasserischool
"""

import numpy as np
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
# Reset the indexes
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

# Problem 5
# Splitting entire dataset into training and test set and shuffling
saveOrgDataset = shuffle(dataset,random_state=0)
multiDataTrain = saveOrgDataset.iloc[:292, :]    # Has first 292 elements
multiDataTest = saveOrgDataset.iloc[292:392,:]   # Has next 100 elements
# Extracting columns 1 - 8 for specific features
multiDataTrainFormat = multiDataTrain.iloc[:,1:8]                
multiDataTestFormat = multiDataTest.iloc[:,1:8]

# Returns a sqaured dataset for the training data
# This function is used purely for formanting a 292x15 dataset
def degreeTwoHelperTrain(df):
    
    df.insert(0, "Bias",biasTrain, True)
    squaredDataset = df**2
    df.insert(8, "CylinderSquared",squaredDataset['Cylinder'],True)
    df.insert(9, "DISPSquared",squaredDataset['Displacement'], True)
    df.insert(10, "HPSquared",squaredDataset['Horsepower'], True)
    df.insert(11, "WTSquared",squaredDataset['Weight'], True)
    df.insert(12, "ACCSquared",squaredDataset['Acceleration'], True)
    df.insert(13, "YRSquared",squaredDataset['Year'], True)
    df.insert(14, "ORGSquared",squaredDataset['Origin'], True)
    
    return df
# Returns a squared dataset for the testing data
# This function is used purely for formanting a 100x15 dataset    
def degreeTwoHelperTest(df):
    
    df.insert(0, "Bias",biasTest, True)
    squaredDataset = df**2
    df.insert(8, "CylinderSquared",squaredDataset['Cylinder'],True)
    df.insert(9, "DISPSquared",squaredDataset['Displacement'], True)
    df.insert(10, "HPSquared",squaredDataset['Horsepower'], True)
    df.insert(11, "WTSquared",squaredDataset['Weight'], True)
    df.insert(12, "ACCSquared",squaredDataset['Acceleration'], True)
    df.insert(13, "YRSquared",squaredDataset['Year'], True)
    df.insert(14, "ORGSquared",squaredDataset['Origin'], True)
    
    return df

# Returns the first 8 columns of the original dataset
def degreeOneHelper(df):
    df = df.iloc[:,:8]
    return df


def multipleLinRegTrain(x,y,degree):
    
    # degree 0
    if (degree == 0):
        # Simply the mean of the mpg values
        degreeZeroTrain = mpg_train.mean()
        degreeZeroTest = mpg_test.mean()
        mseTrain = mean_squared_error(mpg_train,degreeZeroTrain*biasTrain)
        mseTest = mean_squared_error(mpg_test,degreeZeroTest*biasTest)
        return mseTrain,mseTest
    
    # degree 1
    elif (degree == 1):
        # Append ones to the dataset for training values
        addOnes = np.append(arr = np.ones((292, 1)).astype(int), values = x, axis = 1)  
        # Applying the OLS Formula to acquire 1st degree weights
        Xtransposed = np.transpose(addOnes)
        expression1 = np.dot(Xtransposed,addOnes)
        inverse = inv(expression1)
        expression2 = np.dot(Xtransposed,y)
        # degreeOne vector contains the 2nd degree weights
        degreeOne = np.dot(inverse,expression2)

        temp = addOnes
        # Calculating the predicted values based on training data
        pred = degreeOne[0] + degreeOne[1]*temp[:,1] + degreeOne[2]*temp[:,2] + degreeOne[3]*temp[:,3] + degreeOne[4]*temp[:,4] + degreeOne[5]*temp[:,5] + degreeOne[6]*temp[:,6] + degreeOne[7]*temp[:,7]
        # Acquiring the test dataset
        tD = degreeOneHelper(multiDataTestFormat)
        tD.insert(loc=0, column='bias', value=biasTest)
       
        # Calculating the predicted values based on testing data
        pred2 = degreeOne[0] + degreeOne[1]*tD.iloc[:,1] + degreeOne[2]*tD.iloc[:,2] +  degreeOne[3]*tD.iloc[:,3] +  degreeOne[4]*tD.iloc[:,4] +  degreeOne[5]*tD.iloc[:,5] + degreeOne[6]*tD.iloc[:,6] + degreeOne[7]*tD.iloc[:,7] 
        
        # Return both the MSE's for the Training & Testing Data
        return mean_squared_error(y,pred),(mean_squared_error(mpg_test,pred2)) 
    
    elif (degree == 2):
            # Ensuring the proper dimensions for calculations
        if (x.size > 2043 and y.size > 290): 
            # Acquire the 2nd Degree training data                       
            temp = degreeTwoHelperTrain(multiDataTrainFormat)
            # Applying the OLS Formula to acquire 2nd degree weights
            multiTransposed = np.transpose(temp)
            exp1 = np.dot(multiTransposed,temp)
            invert = inv(exp1)
            exp2 = np.dot(multiTransposed,y)
            # degreeTwo vector contains the 2nd degree weights
            degreeTwo = np.dot(invert,exp2)                 
            # b is used to acquire each column of the training data
            b = multiDataTrainFormat
            # Calculating the predicted values based on training data
            ypred = degreeTwo[0] + degreeTwo[1]*b.iloc[:,1] + degreeTwo[2]*b.iloc[:,2] + degreeTwo[3]*b.iloc[:,3] + degreeTwo[4]*b.iloc[:,4] + degreeTwo[5]*b.iloc[:,5] + degreeTwo[6]*b.iloc[:,6] + degreeTwo[7]*b.iloc[:,7] + degreeTwo[8]*b.iloc[:,8] + degreeTwo[9]*b.iloc[:,9] + degreeTwo[10]*b.iloc[:,10] + degreeTwo[11]*b.iloc[:,11] + degreeTwo[12]*b.iloc[:,12] + degreeTwo[13]*b.iloc[:,13] + degreeTwo[14]*b.iloc[:,14]        
            # Calculating the MSE based on the training data
            trainedMSE = mean_squared_error(y,ypred)
            # t is used to acquire each column of the testing data
            t = degreeTwoHelperTest(multiDataTestFormat)
            
            # Calculating the predicted values based on testing data
            ypred2 = degreeTwo[0] + degreeTwo[1]*t.iloc[:,1] + degreeTwo[2]*t.iloc[:,2] + degreeTwo[3]*t.iloc[:,3] + degreeTwo[4]*t.iloc[:,4] + degreeTwo[5]*t.iloc[:,5] + degreeTwo[6]*t.iloc[:,6] + degreeTwo[7]*t.iloc[:,7] + degreeTwo[8]*t.iloc[:,8] + degreeTwo[9]*t.iloc[:,9] + degreeTwo[10]*t.iloc[:,10] + degreeTwo[11]*t.iloc[:,11] + degreeTwo[12]*t.iloc[:,12] + degreeTwo[13]*t.iloc[:,13] + degreeTwo[14]*t.iloc[:,14]        
   
            testedMSE = mean_squared_error(mpg_test,ypred2)
           
            # Return the degreeTwo weights and both the MSE's for the Training & Testing Data
            return degreeTwo, trainedMSE, testedMSE
            


# Printing out the MSE values for 0th,1st, and 2nd order 
print()
mseTrainDegree0 = multipleLinRegTrain(multiDataTrainFormat,mpg_train,0)
mseTestDegree0 = multipleLinRegTrain(multiDataTestFormat,mpg_test,0)
degreeZeroTrain = mseTrainDegree0[0]
degreeZeroTest = mseTrainDegree0[1]
print("Problem 5 MSE 0th Order (Train):",degreeZeroTrain)
print("Problem 5 MSE 0th Order (Test):", degreeZeroTest) 
print() 
mseTrainDegree1 = multipleLinRegTrain(multiDataTrainFormat,mpg_train,1)  
print("Problem 5 MSE 1st Order (Train): ",mseTrainDegree1[0])
print("Problem 5 MSE 1st Order (Test): ",mseTrainDegree1[1])
print()
mseTrainDegree2 = multipleLinRegTrain(multiDataTrainFormat,mpg_train,2)
mseTestDegree2 = multipleLinRegTrain(multiDataTestFormat,mpg_test,2)
mseTrain = mseTrainDegree2[1]
mseTest = mseTrainDegree2[2]
print("Problem 5 MSE 2nd Order (Train): ", mseTrain)   
print("Problem 5 MSE 2nd Order (Test): ", mseTest)
print()


