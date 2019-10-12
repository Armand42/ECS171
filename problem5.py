#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 11 12:08:37 2019

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


# Import the dataset and add labels
dataset = pd.read_csv('auto-mpg.data', delim_whitespace=True, 
                       names=["mpg","Cylinder","Displacement","Horsepower","Weight","Acceleration",
                              "Year","Origin","Model"])
# Remove all ?'s
dataset = dataset.drop(dataset[dataset['Horsepower'] == '?'].index)
# Rest indexes
dataset = dataset.reset_index(drop=True)
#option 2 but this will make Model column Nan
#dataset=dataset.apply(pd.to_numeric, errors='coerce')
dataset['Horsepower'] = dataset['Horsepower'].apply(pd.to_numeric, errors='coerce')


mpg = shuffle(dataset.iloc[:,0].values, random_state = 0)
cyl = shuffle(dataset.iloc[:,1].values, random_state = 0)
disp = shuffle(dataset.iloc[:,2].values, random_state = 0)
hp = shuffle(dataset.iloc[:,3].values, random_state = 0)
wt = shuffle(dataset.iloc[:,4].values,random_state = 0)
acc = shuffle(dataset.iloc[:,5].values, random_state = 0)
yr = shuffle(dataset.iloc[:,6].values, random_state = 0)
org= shuffle(dataset.iloc[:,7].values, random_state = 0)

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


# Problem 4
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
multiDataTrain = saveOrgDataset.iloc[:292, :]     # Has first 292 elements
multiDataTest = saveOrgDataset.iloc[292:392,:]   # Has next 100 elements
# Extracting columns 1 - 8 for specific features
multiDataTrainFormat = multiDataTrain.iloc[:,1:8]                   # Columns 1-8
multiDataTestFormat = multiDataTest.iloc[:,1:8]


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

def degreeOneHelper(df):
    #df = df.iloc[:,1:8]
    #df = df[292:392]
    #df2 = df.insert(0, 'bias', 1)
    
    return df

# Maybe add extra parameter here to specify which dataset we use
def multipleLinRegTrain(x,y,degree):
    
    # degree 0
    # not done yet
    
    if (degree == 1):
        #if (x.size < 2044 and y.size < 292):
            
        #addBiasTest = np.append(arr = np.ones((100, 1)).astype(int), values = x, axis = 1)
        #else:
        addOnes = np.append(arr = np.ones((292, 1)).astype(int), values = x, axis = 1)  # 292 size
        #addBiasTrain = np.append(arr = np.ones((292, 1)).astype(int), values = x, axis = 1)
        
        Xtransposed = np.transpose(addOnes)
        expression1 = np.dot(Xtransposed,addOnes)
        inverse = inv(expression1)
        expression2 = np.dot(Xtransposed,y)
        degreeOne = np.dot(inverse,expression2)
        ###
        temp = addOnes
        pred = degreeOne[0] + degreeOne[1]*temp[:,1] + degreeOne[2]*temp[:,2] + degreeOne[3]*temp[:,3] + degreeOne[4]*temp[:,4] + degreeOne[5]*temp[:,5] + degreeOne[6]*temp[:,6] + degreeOne[7]*temp[:,7]
        
        
        # Something wrong
        firstDegData = dataset
        
        firstDegData = firstDegData.iloc[:,1:8]
        #d = firstDegData.insert(loc=0, column='bias', value=biasTest)
        #tD = shuffle(d.iloc[292:392,:], random_state = 0)
        #print(tD)
        
        #print(tD)
        #print(tD.shape)
        #print(firstDegData)
        
        
     
        
        #pred2 = degreeOne[0] + degreeOne[1]*tD.iloc[:,1] + degreeOne[2]*tD.iloc[:,2] +  degreeOne[3]*tD.iloc[:,3] +  degreeOne[4]*tD.iloc[:,4] +  degreeOne[5]*tD.iloc[:,5] + degreeOne[6]*tD.iloc[:,6] + degreeOne[7]*tD.iloc[:,7] 
        
        #print(mean_squared_error(mpg_test,pred2))
        ####
        
        return mean_squared_error(y,pred)
    
    # Train only dataset 
    elif (degree == 2):
        if (x.size > 2043 and y.size > 290):                        # 292 size PROBLEM HERE 393 instead!!!
            temp = degreeTwoHelperTrain(multiDataTrainFormat)
            #temp = temp.iloc[:292, :]
            multiTransposed = np.transpose(temp)
            exp1 = np.dot(multiTransposed,temp)
            invert = inv(exp1)
            exp2 = np.dot(multiTransposed,y)
            degreeTwo = np.dot(invert,exp2)                 
            
            # Why not just calculate it here for test
            b = multiDataTrainFormat
            
            
            ypred = degreeTwo[0] + degreeTwo[1]*b.iloc[:,1] + degreeTwo[2]*b.iloc[:,2] + degreeTwo[3]*b.iloc[:,3] + degreeTwo[4]*b.iloc[:,4] + degreeTwo[5]*b.iloc[:,5] + degreeTwo[6]*b.iloc[:,6] + degreeTwo[7]*b.iloc[:,7] + degreeTwo[8]*b.iloc[:,8] + degreeTwo[9]*b.iloc[:,9] + degreeTwo[10]*b.iloc[:,10] + degreeTwo[11]*b.iloc[:,11] + degreeTwo[12]*b.iloc[:,12] + degreeTwo[13]*b.iloc[:,13] + degreeTwo[14]*b.iloc[:,14]        
             
            trainedMSE = mean_squared_error(y,ypred)
            
            
            t = degreeTwoHelperTest(multiDataTestFormat)
            
            multiTransposed2 = np.transpose(t)
            exp11 = np.dot(multiTransposed2,t)
            invert2 = inv(exp1)
            exp22 = np.dot(multiTransposed2,mpg_test)
            degreeTwoTwo= np.dot(invert,exp2)  
            
            ypred2 = degreeTwo[0] + degreeTwo[1]*t.iloc[:,1] + degreeTwo[2]*t.iloc[:,2] + degreeTwo[3]*t.iloc[:,3] + degreeTwo[4]*t.iloc[:,4] + degreeTwo[5]*t.iloc[:,5] + degreeTwo[6]*t.iloc[:,6] + degreeTwo[7]*t.iloc[:,7] + degreeTwo[8]*t.iloc[:,8] + degreeTwo[9]*t.iloc[:,9] + degreeTwo[10]*t.iloc[:,10] + degreeTwo[11]*t.iloc[:,11] + degreeTwo[12]*t.iloc[:,12] + degreeTwo[13]*t.iloc[:,13] + degreeTwo[14]*t.iloc[:,14]        
            
            testedMSE = mean_squared_error(mpg_test,ypred2)
            # I really hope this works
            #temp = temp.iloc[:292, :]
            #multiTransposed2 = np.transpose(temp2)
            #exp11 = np.dot(multiTransposed2,temp2)
            #invert2 = inv(exp1)
            #exp22 = np.dot(multiTransposed2,y)
            #degreeTwoTwo = np.dot(invert,exp2)                 
            
            # Why not just calculate it here for test
           # c = multiDataTestFormat
            
            
            return degreeTwo, trainedMSE, testedMSE
            
            
           # return  degreeTwo, mean_squared_error(y,ypred)
        
        # The reason I might not need this is because I am not saving the original weights
       # else: 
        #    temp = degreeTwoHelperTest(multiDataTestFormat)        # 100 size
         #   multiTransposed = np.transpose(temp)
          #  exp1 = np.dot(multiTransposed,temp)
           # invert = inv(exp1)
            #exp2 = np.dot(multiTransposed,y)
            #degreeTwo = np.dot(invert,exp2)
            
           # b = multiDataTestFormat
            #ypred = degreeTwo[0] + degreeTwo[1]*b.iloc[:,1] + degreeTwo[2]*b.iloc[:,2] + degreeTwo[3]*b.iloc[:,3] + degreeTwo[4]*b.iloc[:,4] + degreeTwo[5]*b.iloc[:,5] + degreeTwo[6]*b.iloc[:,6] + degreeTwo[7]*b.iloc[:,7] + degreeTwo[8]*b.iloc[:,8] + degreeTwo[9]*b.iloc[:,9] + degreeTwo[10]*b.iloc[:,10] + degreeTwo[11]*b.iloc[:,11] + degreeTwo[12]*b.iloc[:,12] + degreeTwo[13]*b.iloc[:,13] + degreeTwo[14]*b.iloc[:,14]        

            #mean_squared_error(y,ypred)
            
            #return degreeTwo, mean_squared_error(y,ypred)
        


print()
print("Problem 5 MSE 0th Order (Train): ???")
print("Problem 5 MSE 0th Order (Test): ???") 
print() 
mseTrainDegree1 = multipleLinRegTrain(multiDataTrainFormat,mpg_train,1)  
#mseTestDegree1 = multipleLinRegTrain(multiDataTestFormat,mpg_test,1) 
print("Problem 5 MSE 1st Order (Train): ",mseTrainDegree1)
#print("Problem 5 MSE 1st Order (Test): ",mseTestDegree1)
# Problem 5 Calculated MSE
print()
mseTrainDegree2 = multipleLinRegTrain(multiDataTrainFormat,mpg_train,2)
#mseTestDegree2 = multipleLinRegTrain(multiDataTestFormat,mpg_test,2)

# MSE values in second part
mseTrain = mseTrainDegree2[1]
mseTest = mseTrainDegree2[2]
#mseTest = mseTestDegree2[1]
#ypred = mseTrain[0] + mseTrain[1]*multiDataTestFormat.iloc[:,1] + mseTrain[2]*multiDataTestFormat.iloc[:,2] + mseTrain[3]*multiDataTestFormat.iloc[:,3] + mseTrain[4]*multiDataTestFormat.iloc[:,4] + mseTrain[5]*multiDataTestFormat.iloc[:,5] + mseTrain[6]*multiDataTestFormat.iloc[:,6] + mseTrain[7]*multiDataTestFormat.iloc[:,7] + mseTrain[8]*multiDataTestFormat.iloc[:,8] + mseTrain[9]*multiDataTestFormat.iloc[:,9] + mseTrain[10]*multiDataTestFormat.iloc[:,10] + mseTrain[11]*multiDataTestFormat.iloc[:,11] + mseTrain[12]*multiDataTestFormat.iloc[:,12] + mseTrain[13]*multiDataTestFormat.iloc[:,13] + mseTrain*multiDataTestFormat.iloc[:,14]        


print("Problem 5 MSE 2nd Order (Train): ", mseTrain)   # change back after
print("Problem 5 MSE 2nd Order (Test): ", mseTest)
print()

#lol = degreeTwoHelperTest2(multiDataTestFormat)
#mseTrain = mseTrainDegree2[0]

#ypred = mseTrain[0] + mseTrain[1]*lol.iloc[:,1] + mseTrain[2]*lol.iloc[:,2] + mseTrain[3]*lol.iloc[:,3] + mseTrain[4]*lol.iloc[:,4] + mseTrain[5]*lol.iloc[:,5] + mseTrain[6]*lol.iloc[:,6] + mseTrain[7]*lol.iloc[:,7] + mseTrain[8]*lol.iloc[:,8] + mseTrain[9]*lol.iloc[:,9] + mseTrain[10]*lol.iloc[:,10] + mseTrain[11]*lol.iloc[:,11] + mseTrain[12]*lol.iloc[:,12] + mseTrain[13]*lol.iloc[:,13] + mseTrain*lol.iloc[:,14]        


#m = mseTrainDegree2[2]    
#print(m)     

# THIS WORKSSSSSSSS WOOOOOHOOOOO NOW FIX THE OTHER ONES
    
weeeeeee = degreeOneHelper(dataset)