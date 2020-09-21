#!/usr/bin/env python
# coding: utf-8

# # L2 Generalized Linear Regression

# In[137]:


import pandas as pd
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import KFold # import KFold
from sklearn.model_selection import cross_val_score


# In[138]:


def ridge_regression(data, predictors, alpha):
    #Fit the model
    ridgereg = Ridge(alpha=alpha,normalize=True)
    ridgereg.fit(data[predictors],data['y'])
    y_pred = ridgereg.predict(data[predictors])
    
    #Check if a plot is to be made for the entered alpha
    #if alpha in models_to_plot:
    #    plt.subplot(models_to_plot[alpha])
    #    plt.tight_layout()
    #    plt.plot(data['x'],y_pred)
    #    plt.plot(data['x'],data['y'],'.')
    #    plt.title('Plot for alpha: %.3g'%alpha)
    
    #Return the result in pre-defined format
    rss = sum((y_pred-data['y'])**2)
    mse = rss/len(data)
    ret = mse
    #ret.extend([ridgereg.intercept_])
    #ret.extend(ridgereg.coef_)
    return ret


# In[22]:


initial = 0
alpha = []

while (initial <= 150):
    alpha.append(initial)
    initial = initial + 1

predictors=[]
predictors.extend(['x%d'%i for i in range(1,101)])
predictors1 = []
predictors1.extend(['x%d'%i for i in range(1,11)])


# In[15]:


data100 = pd.read_csv('train-100-100.csv')
ridge_regression(data100, predictors, 2)


# In[36]:


def ridge_regression_joint(data, alpha):
    MSE = []
    for i in alpha:
        MSE.append(ridge_regression(data, predictors, i))
    return MSE
def ridge_regression_joint1(data, alpha):
    MSE = []
    for i in alpha:
        MSE.append(ridge_regression(data, predictors1, i))
    return MSE


# In[51]:


#Plot 1 MSE - train-100-100.csv
train_100_100 = pd.read_csv('train-100-100.csv')
plot1_MSE = ridge_regression_joint(train_100_100, alpha)

#Plot 2 MSE - train-1000-100.csv
train_1000_100 = pd.read_csv('train-1000-100.csv')
plot2_MSE = ridge_regression_joint(train_1000_100, alpha)

#Plot 3 MSE - train-50(1000)-100.csv
#train_50_1000_100 = pd.read_csv('train-50(1000)-100.csv')
train_50_1000_100 = train_1000_100.head(50)
plot3_MSE = ridge_regression_joint(train_50_1000_100, alpha)

#Plot 4 MSE - train-100(1000)-100.csv
#train_100_1000_100 = pd.read_csv('train-100(1000)-100.csv')
train_100_1000_100 = train_1000_100.head(100)
plot4_MSE = ridge_regression_joint(train_100_1000_100, alpha)

#Plot 5 MSE - train-150(1000)-100.csv
#train_150_1000_100 = pd.read_csv('train-150(1000)-100.csv')
train_150_1000_100 = train_1000_100.head(150)
plot5_MSE = ridge_regression_joint(train_150_1000_100, alpha)

#Plot 6 MSE - test-100-10.csv
train_100_10 = pd.read_csv('train-100-10.csv')
plot6_MSE = ridge_regression_joint1(train_100_10, alpha)

#Plot 7 MSE - test-100-100.csv
test_100_100 = pd.read_csv('test-100-100.csv')
plot7_MSE = ridge_regression_joint(test_100_100, alpha)

#Plot 8 MSE - test-1000-100.csv
test_1000_100 = pd.read_csv('test-1000-100.csv')
plot8_MSE = ridge_regression_joint(test_1000_100, alpha)

#Plot 9 MSE - test-50(1000)-100.csv
#train_50_1000_100 = pd.read_csv('train-50(1000)-100.csv')
test_50_1000_100 = test_1000_100.head(50)
plot9_MSE = ridge_regression_joint(test_50_1000_100, alpha)

#Plot 10 MSE - test-100(1000)-100.csv
#train_100_1000_100 = pd.read_csv('train-100(1000)-100.csv')
test_100_1000_100 = test_1000_100.head(100)
plot10_MSE = ridge_regression_joint(test_100_1000_100, alpha)

#Plot 11 MSE - test-150(1000)-100.csv
#train_150_1000_100 = pd.read_csv('train-150(1000)-100.csv')
test_150_1000_100 = test_1000_100.head(150)
plot11_MSE = ridge_regression_joint(test_150_1000_100, alpha)

#Plot 12 MSE - test-100-10.csv
test_100_10 = pd.read_csv('test-100-10.csv')
plot12_MSE = ridge_regression_joint1(test_100_10, alpha)


# In[40]:


df = pd.DataFrame({'x':alpha, 'p1': plot1_MSE, 'p2': plot2_MSE, 'p3': plot3_MSE, 'p4': plot4_MSE, 'p5': plot5_MSE, 'p6': plot6_MSE, 'p7': plot7_MSE, 'p8': plot8_MSE, 'p9': plot9_MSE, 'p10': plot10_MSE, 'p11': plot11_MSE, 'p12': plot12_MSE })


# In[47]:


# multiple line plot
plt.plot( 'x', 'p1', data=df, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=2, label = "train-100-100")
plt.plot( 'x', 'p2', data=df, marker='*', color='olive', linewidth=2, label = "train-1000-100")
plt.plot( 'x', 'p3', data=df, marker='', color='olive', linewidth=2, linestyle='dashed', label="train-50(1000)-100")
plt.plot( 'x', 'p4', data=df, marker='o', markerfacecolor='olive', markersize=12, color='skyblue', linewidth=2, label = "train-100(1000)-100")
plt.plot( 'x', 'p5', data=df, marker='*', color='red', linewidth=2, label = "train-150(1000)-100")
plt.plot( 'x', 'p6', data=df, marker='', color='red', linewidth=2, linestyle='dashed', label="train-100-10")
plt.plot( 'x', 'p7', data=df, marker='o', markerfacecolor='yellow', markersize=12, color='skyblue', linewidth=2, label = "test-100-100")
plt.plot( 'x', 'p8', data=df, marker='*', color='yellow', linewidth=2, label = "test-1000-100")
plt.plot( 'x', 'p9', data=df, marker='', color='yellow', linewidth=2, linestyle='dashed', label="test-50(1000)-100")
plt.plot( 'x', 'p10', data=df, marker='o', markerfacecolor='green', markersize=12, color='skyblue', linewidth=2, label = "test-100(1000)-100")
plt.plot( 'x', 'p11', data=df, marker='*', color='green', linewidth=2, label = "test-150(1000)-100")
plt.plot( 'x', 'p12', data=df, marker='', color='green', linewidth=2, linestyle='dashed', label="test-100-10")
plt.legend()


# In[48]:


df


# In[50]:


initial = 1
alpha1 = []

while (initial <= 150):
    alpha1.append(initial)
    initial = initial + 1


# In[64]:


#Plot 1 MSE - train-100-100.csv
train_100_100 = pd.read_csv('train-100-100.csv')
plot1_MSE = ridge_regression_joint(train_100_100, alpha1)

#Plot 2 MSE - train-50(1000)-100.csv
#train_50_1000_100 = pd.read_csv('train-50(1000)-100.csv')
train_50_1000_100 = train_1000_100.head(50)
plot2_MSE = ridge_regression_joint(train_50_1000_100, alpha1)

#Plot 3 MSE - train-100(1000)-100.csv
#train_100_1000_100 = pd.read_csv('train-100(1000)-100.csv')
train_100_1000_100 = train_1000_100.head(100)
plot3_MSE = ridge_regression_joint(train_100_1000_100, alpha1)

#Plot 4 MSE - test-100-100.csv
test_100_100 = pd.read_csv('test-100-100.csv')
plot4_MSE = ridge_regression_joint(test_100_100, alpha1)

#Plot 5 MSE - test-50(1000)-100.csv
#train_50_1000_100 = pd.read_csv('train-50(1000)-100.csv')
test_50_1000_100 = test_1000_100.head(50)
plot5_MSE = ridge_regression_joint(test_50_1000_100, alpha1)

#Plot 6 MSE - test-100(1000)-100.csv
#train_100_1000_100 = pd.read_csv('train-100(1000)-100.csv')
test_100_1000_100 = test_1000_100.head(100)
plot6_MSE = ridge_regression_joint(test_100_1000_100, alpha1)


# In[65]:


df1 = pd.DataFrame({'x':alpha1, 'p1': plot1_MSE, 'p2': plot2_MSE, 'p3': plot3_MSE, 'p4': plot4_MSE, 'p5': plot5_MSE, 'p6': plot6_MSE })

# multiple line plot
plt.plot( 'x', 'p1', data=df1, marker='o', markerfacecolor='blue', markersize=12, color='skyblue', linewidth=2, label = "train-100-100")
plt.plot( 'x', 'p2', data=df1, marker='*', color='olive', linewidth=2, label = "train-50(1000)-100")
plt.plot( 'x', 'p3', data=df1, marker='', color='olive', linewidth=2, linestyle='dashed', label="train-100(1000)-100")
plt.plot( 'x', 'p4', data=df1, marker='o', markerfacecolor='olive', markersize=12, color='skyblue', linewidth=2, label = "test-100-100")
plt.plot( 'x', 'p5', data=df1, marker='*', color='red', linewidth=2, label = "test-50(1000)-100")
plt.plot( 'x', 'p6', data=df1, marker='', color='red', linewidth=2, linestyle='dashed', label="train-1000(1000)-100")
plt.legend()


# In[83]:


#X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]]) # create an array
#y = np.array([1, 2, 3, 4]) # Create another array

X = train_1000_100[["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20","x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30","x31", "x32", "x33", "x34", "x35", "x36", "x37", "x38", "x39", "x40","x41", "x42", "x43", "x44", "x45", "x46", "x47", "x48", "x49", "x50","x51", "x52", "x53", "x54", "x55", "x56", "x57", "x58", "x59", "x60","x61", "x62", "x63", "x64", "x65", "x66", "x67", "x68", "x69", "x70","x71", "x72", "x73", "x74", "x75", "x76", "x77", "x78", "x79", "x80","x81", "x82", "x83", "x84", "x85", "x86", "x87", "x88", "x89", "x90","x91", "x92", "x93", "x94", "x95", "x96", "x97", "x98", "x99","x100"]].to_numpy()
y = train_1000_100[["y"]]
kf = KFold(n_splits=10) # Define the split - into 2 folds 
kf.get_n_splits(X) # returns the number of splitting iterations in the cross-validator


ridgereg = Ridge(alpha=1,normalize=True)
#ridgereg.fit(data[predictors],data['y'])
scores = cross_val_score(ridgereg, X, y,scoring='neg_mean_squared_error', cv=10)
    
#clf = svm.SVC(kernel='linear', C=1)
#scores = cross_val_score(clf, x, y, cv=5)
MSE = scores*(-1)
np.mean(MSE)


# In[128]:


def cross_val(data, alpha):
    X = data[["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10", "x11", "x12", "x13", "x14", "x15", "x16", "x17", "x18", "x19", "x20","x21", "x22", "x23", "x24", "x25", "x26", "x27", "x28", "x29", "x30","x31", "x32", "x33", "x34", "x35", "x36", "x37", "x38", "x39", "x40","x41", "x42", "x43", "x44", "x45", "x46", "x47", "x48", "x49", "x50","x51", "x52", "x53", "x54", "x55", "x56", "x57", "x58", "x59", "x60","x61", "x62", "x63", "x64", "x65", "x66", "x67", "x68", "x69", "x70","x71", "x72", "x73", "x74", "x75", "x76", "x77", "x78", "x79", "x80","x81", "x82", "x83", "x84", "x85", "x86", "x87", "x88", "x89", "x90","x91", "x92", "x93", "x94", "x95", "x96", "x97", "x98", "x99","x100"]].to_numpy()
    y = data[["y"]]
    ridgereg = Ridge(alpha=alpha,normalize=True)
    #ridgereg.fit(data[predictors],data['y'])
    scores = cross_val_score(ridgereg, X, y,scoring='neg_mean_squared_error', cv=10)
    MSE = scores*(-1)
    AVG_MSE = np.mean(MSE)
    return AVG_MSE
    


# In[129]:


#cv_result_100_10 = []

#for i in alpha:
#    cv_result_100_10.append(cross_val(train_1000_100, i))

def Find_Least_MSE_Lambda(data):
    result = []
    min_lambda = 0
    for i in alpha:
        if len(result) != 0:
            minimum = min(result)
        else:
            minimum = 0
        
        res = cross_val(data, i)
        result.append(res)
        if res > minimum:
            min_lambda = min_lambda
        else:
            min_lamda = i
    return min_lambda, result 


# In[133]:


def cross_val1(data, alpha):
    X = data[["x1", "x2", "x3", "x4", "x5", "x6", "x7", "x8", "x9", "x10"]]
    y = data[["y"]]
    ridgereg = Ridge(alpha=alpha,normalize=True)
    #ridgereg.fit(data[predictors],data['y'])
    scores = cross_val_score(ridgereg, X, y,scoring='neg_mean_squared_error', cv=10)
    MSE = scores*(-1)
    AVG_MSE = np.mean(MSE)
    return AVG_MSE
def Find_Least_MSE_Lambda1(data):
    result = []
    min_lambda = 0
    for i in alpha:
        if len(result) != 0:
            minimum = min(result)
        else:
            minimum = 0
        
        res = cross_val1(data, i)
        result.append(res)
        if res > minimum:
            min_lambda = min_lambda
        else:
            min_lamda = i
    return min_lambda, result 


# In[135]:


lambda_optimum_100_100, result_100_100 = Find_Least_MSE_Lambda(train_100_100)
lambda_optimum_1000_100, result_1000_100 = Find_Least_MSE_Lambda(train_1000_100)
lambda_optimum_50_1000_100, result_50_1000_100 = Find_Least_MSE_Lambda(train_50_1000_100)
lambda_optimum_100_1000_100, result_100_1000_100 = Find_Least_MSE_Lambda(train_100_1000_100)
lambda_optimum_150_1000_100, result_150_1000_100 = Find_Least_MSE_Lambda(train_150_1000_100)
lambda_optimum_100_10, result_100_10 = Find_Least_MSE_Lambda1(train_100_10)


# In[136]:


print('Set_100_100 Optimum Lambda - ', lambda_optimum_100_100)
print('Set_1000_100 Optimum Lambda - ', lambda_optimum_1000_100)
print('Set_50(1000)_100 Optimum Lambda - ', lambda_optimum_50_1000_100)
print('Set_100(1000)_100 Optimum Lambda - ', lambda_optimum_100_1000_100)
print('Set_150(1000)_100 Optimum Lambda - ', lambda_optimum_150_1000_100)
print('Set_100_100 Optimum Lambda - ', lambda_optimum_100_10)


# In[134]:


lambda_optimum_100_10, result_100_10 = Find_Least_MSE_Lambda1(train_100_10)

