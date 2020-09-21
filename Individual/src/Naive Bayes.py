#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB


# In[6]:


train = pd.read_csv('problem1_train.csv')
test = pd.read_csv('problem1_test.csv')


# In[13]:


X_train = np.array(train[['Education Level','Career','Years of Experience']])
Y_train = np.array(train[['Salary']])
X_train = np.array(test[['Education Level','Career','Years of Experience']])
Y_train = np.array(test[['Salary']])


# In[14]:


train = pd.read_csv('train_matrix.csv')
test = pd.read_csv('test_matrix.csv')


# In[23]:


X_train = np.array(train[['Education Level','Career','Years of Experience']])
Y_train = np.array(train[['Salary']])
X_test = np.array(test[['Education Level','Career','Years of Experience']])
Y_test = np.array(test[['Salary']])


# In[24]:


nbmodel = GaussianNB()
nbmodel.fit(X_train, Y_train)


# In[25]:


nbmodel.predict(X_test)


# #### The predictions are the following -
# 1. High School Service Less than 3   --- Low
# 2. College Retail Less than 3 --- High
# 3. Graduate Service 3 to 10 --- Low

# In[ ]:




