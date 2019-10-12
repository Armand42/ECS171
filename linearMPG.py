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




#dataset = pd.read_csv('auto-mpg.data', delim_whitespace=True)
# Store each column into separate independent variables


# ENSURE THIS WORKS AT ALLLLLLLLLLLLL COSTTSSSSS
# Shuffle dataset before splitting

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
print("low mpg =",low,"med mpg =",med,"high mpg =",high, "very high mpg =",vhigh)

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

# Problem 3 Regressor
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



# COMMENT OUT SNS PAIRPLOT option 2 to see individual graphs


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
    #print("Oth Order MSE: ???")
    #print("1st Order MSE (Training):",mean_squared_error(y,w0+x*w1)) 
    #print("2nd Order MSE (Training):",mean_squared_error(y,w00+x*w11 +w22*x**2)) 
    #print("3rd Order MSE (Training):",mean_squared_error(y,w000+x*w111+ w222*x**2 + w333*x**3)) 
    
    
   


    #plt.plot(x,w0+x*w1, color = "red")
    #plt.plot(x,w00+x*w11 +w22*x**2, color = "blue")
    #plt.plot(x,w000+x*w111+ w222*x**2 + w333*x**3, color = "green")
    #plt.title('(Training set)')
    #plt.xlabel('input')
    #plt.ylabel('mpg')
    #plt.show()
    
    return degreeZero, degreeOne, degreeTwo, degreeThree

def plotTrainData(xtrain, ytrain):
    
    # Takes in the return values of the training data
    data = linReg(xtrain, ytrain)
    
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
    
    print("Oth Order MSE: (Training):", mean_squared_error(ytrain,degreeZero*biasTrain))
    print("1st Order MSE (Training):",mean_squared_error(ytrain,w0+xtrain*w1)) 
    print("2nd Order MSE (Training):",mean_squared_error(ytrain,w00+xtrain*w11 +w22*xtrain**2)) 
    print("3rd Order MSE (Training):",mean_squared_error(ytrain,w000+xtrain*w111+ w222*xtrain**2 + w333*xtrain**3)) 
    
    # Sort for ensuring correct regression displays
    xtrain = np.sort(xtrain)
    ytrain = np.sort(ytrain)
    
    plt.plot(xtrain,w0+xtrain*w1, color = "red")
    plt.plot(xtrain,w00+xtrain*w11 +w22*xtrain**2, color = "blue")
    plt.plot(xtrain,w000+xtrain*w111+ w222*xtrain**2 + w333*xtrain**3, color = "green")
    plt.title('(Training set)')
    plt.xlabel('input')
    plt.ylabel('mpg')
    
    plt.show()
    

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
    print("Oth Order MSE (Testing):", mean_squared_error(ytest,degreeZero*biasTest))
    print("1st Order MSE (Testing):",mean_squared_error(ytest,w0+xtest*w1)) 
    print("2nd Order MSE (Testing):",mean_squared_error(ytest,w00+xtest*w11 +w22*xtest**2)) 
    print("3rd Order MSE (Testing):",mean_squared_error(ytest,w000+xtest*w111+ w222*xtest**2 + w333*xtest**3))
   
    # Sort for ensuring correct regression displays
    xtest = np.sort(xtest)
    ytest = np.sort(ytest)
    
    plt.axhline(y=degreeZero, color='y')
    plt.plot(xtest,w0+xtest*w1, color = "red")
    plt.plot(xtest,w00+xtest*w11 +w22*xtest**2, color = "blue")
    plt.plot(xtest,w000+xtest*w111+ w222*xtest**2 + w333*xtest**3, color = "green")
    plt.title('(Testing set)')
    plt.xlabel('input variable')
    plt.ylabel('mpg')
    plt.show()
    
# Issues:
    
    # Are my MSEs correct?
    # How to calculate 0th order MSE
    
    
# How to do 5 and 6
# Dont forget that you commented out the big chart and testers!!!

# Plotting Train and Test Regressions

print("Cylinder MSE:")
trainReg1 = plotTrainData(cyl_train,mpg_train)
print("Cylinder MSE:")
testReg1 = plotTestData(cyl_test, mpg_test)

print("Displacement MSE:")
trainReg2 = plotTrainData(disp_train,mpg_train)
print("Displacement MSE:")
testReg2 = plotTestData(disp_test, mpg_test)

print("Horsepower MSE:")
trainReg3 = plotTrainData(hp_train,mpg_train)
print("Horsepower MSE:")
testReg3 = plotTestData(hp_test, mpg_test)

print("Weight MSE:")
trainReg4 = plotTrainData(wt_train,mpg_train)
print("Weight MSE:")
testReg4 = plotTestData(wt_test, mpg_test)

print("Acceleration MSE:")
trainReg5 = plotTrainData(acc_train,mpg_train)
print("Acceleration MSE:")
testReg5 = plotTestData(acc_test, mpg_test)

print("Year MSE:")
trainReg6 = plotTrainData(yr_train,mpg_train)
print("Year MSE:")
testReg6 = plotTestData(yr_test, mpg_test)

print("Origin MSE:")
trainReg7 = plotTrainData(org_train,mpg_train)
print("Origin MSE:")
testReg7 = plotTestData(org_test, mpg_test)



# Problem 5
# Splitting entire dataset into training and test set and shuffling
saveOrgDataset = shuffle(dataset,random_state=0)
multiDataTrain = saveOrgDataset.iloc[:292, :]     # Has first 292 elements
multiDataTest = saveOrgDataset.iloc[292:392,:]   # Has next 100 elements
# Extracting columns 1 - 8 for specific features
multiDataTrainFormat = multiDataTrain.iloc[:,1:8]                   # Columns 1-8
multiDataTestFormat = multiDataTest.iloc[:,1:8]

# Formatting Training Dataset for 2nd Degree

# x = dataset
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

# Formatting Testing Dataset for 2nd Degree
#multiDataTestFormat.insert(0, "Bias",biasTest, True)
#squaredDataset2 = multiDataTestFormat**2
#multiDataTestFormat.insert(8, "CylinderSquared",squaredDataset2['Cylinder'],True)
#multiDataTestFormat.insert(9, "DISPSquared",squaredDataset2['Displacement'], True)
#multiDataTestFormat.insert(10, "HPSquared",squaredDataset2['Horsepower'], True)
#multiDataTestFormat.insert(11, "WTSquared",squaredDataset2['Weight'], True)
#multiDataTestFormat.insert(12, "ACCSquared",squaredDataset2['Acceleration'], True)
#multiDataTestFormat.insert(13, "YRSquared",squaredDataset2['Year'], True)
#multiDataTestFormat.insert(14, "ORGSquared",squaredDataset2['Origin'], True)


# Maybe add extra parameter here to specify which dataset we use
def multipleLinRegTrain(x,y,degree):
    
    # degree 0
    # not done yet
    
    if (degree == 1):
        if (x.size < 2044 and y.size < 292):
            addOnes = np.append(arr = np.ones((100, 1)).astype(int), values = x, axis = 1)  # 100 size
        #addBiasTest = np.append(arr = np.ones((100, 1)).astype(int), values = x, axis = 1)
        else:
            addOnes = np.append(arr = np.ones((292, 1)).astype(int), values = x, axis = 1)  # 292 size
        #addBiasTrain = np.append(arr = np.ones((292, 1)).astype(int), values = x, axis = 1)
        
        Xtransposed = np.transpose(addOnes)
        expression1 = np.dot(Xtransposed,addOnes)
        inverse = inv(expression1)
        expression2 = np.dot(Xtransposed,y)
        degreeOne = np.dot(inverse,expression2)
        
        temp = addOnes
        
        pred = degreeOne[0] + degreeOne[1]*temp[:,1] + degreeOne[2]*temp[:,2] +  degreeOne[3]*temp[:,3] +  degreeOne[4]*temp[:,4] +  degreeOne[5]*temp[:,5] + degreeOne[6]*temp[:,6] + + degreeOne[7]*temp[:,7]
        
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
            
            
            mean_squared_error(y,ypred)
            
            return  degreeTwo, mean_squared_error(y,ypred)
        
        # The reason I might not need this is because I am not saving the original weights
        else: 
            temp = degreeTwoHelperTest(multiDataTestFormat)        # 100 size
            multiTransposed = np.transpose(temp)
            exp1 = np.dot(multiTransposed,temp)
            invert = inv(exp1)
            exp2 = np.dot(multiTransposed,y)
            degreeTwo = np.dot(invert,exp2)
            
            b = multiDataTestFormat
            ypred = degreeTwo[0] + degreeTwo[1]*b.iloc[:,1] + degreeTwo[2]*b.iloc[:,2] + degreeTwo[3]*b.iloc[:,3] + degreeTwo[4]*b.iloc[:,4] + degreeTwo[5]*b.iloc[:,5] + degreeTwo[6]*b.iloc[:,6] + degreeTwo[7]*b.iloc[:,7] + degreeTwo[8]*b.iloc[:,8] + degreeTwo[9]*b.iloc[:,9] + degreeTwo[10]*b.iloc[:,10] + degreeTwo[11]*b.iloc[:,11] + degreeTwo[12]*b.iloc[:,12] + degreeTwo[13]*b.iloc[:,13] + degreeTwo[14]*b.iloc[:,14]        

            mean_squared_error(y,ypred)
            
            return degreeTwo, mean_squared_error(y,ypred)
        

def multipleLinRegTest(xtest,ytest,degree):
    # degree 0
    
    # degree 2
    if (degree == 2):
        
        values = multipleLinRegTrain(xtest,ytest,degree)
        w00 = values[0][0]
        w01 = values[0][1]
        w02 = values[0][2]
        w03 = values[0][3]
        w04 = values[0][4]
        w05 = values[0][5]
        w06 = values[0][6]
        w07 = values[0][7]
        w08 = values[0][8]
        w09 = values[0][9]
        w10 = values[0][10]
        w11 = values[0][11]
        w12 = values[0][12]
        w13 = values[0][13]
        w14 = values[0][14]
        
        ypred = w00 + w01*xtest.iloc[:,1] + w02*xtest.iloc[:,2] + w03*xtest.iloc[:,3] + w04*xtest.iloc[:,4] + w05*xtest.iloc[:,5] + w06*xtest.iloc[:,6] + w07*xtest.iloc[:,7] + w08*xtest.iloc[:,8] + w09*xtest.iloc[:,9] + w10*xtest.iloc[:,10] + w11*xtest.iloc[:,11] + w12*xtest.iloc[:,12] + w13*xtest.iloc[:,13] + w14*xtest.iloc[:,14]        

        return mean_squared_error(ytest,ypred)


print()
print("Problem 5 MSE 0th Order (Train): ???")
print("Problem 5 MSE 0th Order (Test): ???") 
print() 
mseTrainDegree1 = multipleLinRegTrain(multiDataTrainFormat,mpg_train,1)  
mseTestDegree1 = multipleLinRegTrain(multiDataTestFormat,mpg_test,1) 
print("Problem 5 MSE 1st Order (Train): ",mseTrainDegree1)
print("Problem 5 MSE 1st Order (Test): ",mseTestDegree1)
# Problem 5 Calculated MSE
print()
mseTrainDegree2 = multipleLinRegTrain(multiDataTrainFormat,mpg_train,2)
mseTestDegree2 = multipleLinRegTrain(multiDataTestFormat,mpg_test,2)

# MSE values in second part
mseTrain = mseTrainDegree2[1]
mseTest = mseTestDegree2[1]
#ypred = mseTrain[0] + mseTrain[1]*multiDataTestFormat.iloc[:,1] + mseTrain[2]*multiDataTestFormat.iloc[:,2] + mseTrain[3]*multiDataTestFormat.iloc[:,3] + mseTrain[4]*multiDataTestFormat.iloc[:,4] + mseTrain[5]*multiDataTestFormat.iloc[:,5] + mseTrain[6]*multiDataTestFormat.iloc[:,6] + mseTrain[7]*multiDataTestFormat.iloc[:,7] + mseTrain[8]*multiDataTestFormat.iloc[:,8] + mseTrain[9]*multiDataTestFormat.iloc[:,9] + mseTrain[10]*multiDataTestFormat.iloc[:,10] + mseTrain[11]*multiDataTestFormat.iloc[:,11] + mseTrain[12]*multiDataTestFormat.iloc[:,12] + mseTrain[13]*multiDataTestFormat.iloc[:,13] + mseTrain*multiDataTestFormat.iloc[:,14]        


print("Problem 5 MSE 2nd Order (Train): ", mseTrain)
print("Problem 5 MSE 2nd Order (Test): ", mseTest)
print()



# Problem 6 - What values are we regressing on

# Problem 7

# Problem 8 - Will use return value from above
print("Problem 8\n")
weight = mseTestDegree2[0]
prediction = weight[0] + weight[1]*4 + weight[2]*400 + weight[3]*150 +weight[4]*3500 + weight[5]*8 + weight[6]*81 + weight[7]*1 + weight[8]*4**2 + weight[9]*400**2 + weight[10]*150**2 + weight[11]*3500**2 + weight[12]*8**2 + weight[13]*81**2 +weight[14]*1**2
print("The predicted MPG rating for 2nd Order Multivariate Regression is:", prediction)

print("The predicted MPG rating for Logistic Regression is: ???")


#ypred = mseTrain[0][0] + mseTrain[0][1]*lol.iloc[:,1] + mseTrain[0][2]*lol.iloc[:,2] + mseTrain[0][3]*lol.iloc[:,3] + mseTrain[0][4]*lol.iloc[:,4] + mseTrain[0][5]*lol.iloc[:,5] + mseTrain[0][6]*lol.iloc[:,6] + mseTrain[0][7]*lol.iloc[:,7] + mseTrain[0][8]*lol.iloc[:,8] + mseTrain[0][9]*lol.iloc[:,9] + mseTrain[0][10]*lol.iloc[:,10] + mseTrain[0][11]*lol.iloc[:,11] + mseTrain[0][12]*lol.iloc[:,12] + mseTrain[0][13]*lol.iloc[:,13] + mseTrain[0][14].lol.iloc[:,14]        
