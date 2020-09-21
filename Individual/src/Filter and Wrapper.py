#!/usr/bin/env python
# coding: utf-8

# In[96]:


get_ipython().system('pip install mlxtend')


# In[98]:


from scipy.io import arff
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


from numpy.random import randn
from numpy.random import seed
from numpy import cov
from sklearn.model_selection import train_test_split, LeaveOneOut, KFold, cross_val_score

from sklearn import model_selection

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut
from mlxtend.feature_selection import SequentialFeatureSelector


# In[3]:


df = arff.loadarff('veh-prime.arff')
df = pd.DataFrame(df[0])

df.head()


# In[18]:


df['target'] = np.where(df['CLASS'] == b'car',1,0)


# In[19]:


df.head()


# In[24]:


pearsonr(df['f0'], df['target'])


# In[25]:


pearsonr(df['f1'], df['target'])


# In[26]:


pearsonr(df['f2'], df['target'])


# In[27]:


pearsonr(df['f3'], df['target'])


# In[28]:


pearsonr(df['f4'], df['target'])


# In[29]:


pearsonr(df['f5'], df['target'])


# In[30]:


pearsonr(df['f6'], df['target'])


# ### Filter Method

# In[44]:


i = 0
corr = []
features = [] 

while (i <= 35):
    feature = 'f' + str(i)
    features.append(feature)
    correlation,_ = pearsonr(df[feature], df['target'])
    corr.append(abs(correlation))
    i = i + 1
final = pd.DataFrame({'Features': features, 'Correlation': corr})
final


# ##### We have considerd the absolute r because we need to find the features which does not have any correlation. As we need to sort the features as per the correlation, the features with high negative correlation will come in our list which is not correct. Hence, we need to calculate the absolute values of the correlation r.

# In[45]:


final = final.sort_values(by = 'Correlation', ascending=True)


# In[46]:


final


# In[48]:


final_features = final[final['Correlation'] <= 0.2]


# In[50]:


final_features['Features']


# ##### We will be considering the above 25 features in our model - f5, f12, f23, f24, f30, f3, f9, f27, f18, f29, f35, f15, f6, f33, f11, f21, f10, f0, f8, f32, f17, f19, f25, f28, f2. 
# 

# In[83]:


accuracy = []
m_features = []

m = 25
X = np.array(df[['f5', 'f12', 'f23', 'f24', 'f30', 'f3', 'f9', 'f27', 'f18', 'f29', 'f35', 'f15', 'f6', 'f33', 'f11', 'f21', 'f10', 'f0', 'f8', 'f32', 'f17', 'f19', 'f25', 'f28', 'f2']])
Y = np.array(df[['target']])

loocv = model_selection.LeaveOneOut()
knn_loocv = KNeighborsClassifier(n_neighbors=7)
results_loocv = model_selection.cross_val_score(knn_loocv, X, Y, cv=loocv)

m_features.append(25)
accuracy.append(results_loocv.mean()*100.0)


# In[84]:



X = np.array(df[['f5', 'f12', 'f23', 'f24', 'f30', 'f3', 'f9', 'f27', 'f18', 'f29', 'f35', 'f15', 'f6', 'f33', 'f11', 'f21', 'f10', 'f0', 'f8', 'f32', 'f17', 'f19', 'f25', 'f28']])
Y = np.array(df[['target']])

loocv = model_selection.LeaveOneOut()
knn_loocv = KNeighborsClassifier(n_neighbors=7)
results_loocv = model_selection.cross_val_score(knn_loocv, X, Y, cv=loocv)

m_features.append(24)
accuracy.append(results_loocv.mean()*100.0)


# In[85]:


X = np.array(df[['f5', 'f12', 'f23', 'f24', 'f30', 'f3', 'f9', 'f27', 'f18', 'f29', 'f35', 'f15', 'f6', 'f33', 'f11', 'f21', 'f10', 'f0', 'f8', 'f32', 'f17', 'f19', 'f25']])
Y = np.array(df[['target']])

loocv = model_selection.LeaveOneOut()
knn_loocv = KNeighborsClassifier(n_neighbors=7)
results_loocv = model_selection.cross_val_score(knn_loocv, X, Y, cv=loocv)

m_features.append(23)
accuracy.append(results_loocv.mean()*100.0)


# In[86]:


X = np.array(df[['f5', 'f12', 'f23', 'f24', 'f30', 'f3', 'f9', 'f27', 'f18', 'f29', 'f35', 'f15', 'f6', 'f33', 'f11', 'f21', 'f10', 'f0', 'f8', 'f32', 'f17', 'f19']])
Y = np.array(df[['target']])

loocv = model_selection.LeaveOneOut()
knn_loocv = KNeighborsClassifier(n_neighbors=7)
results_loocv = model_selection.cross_val_score(knn_loocv, X, Y, cv=loocv)

m_features.append(22)
accuracy.append(results_loocv.mean()*100.0)


# In[82]:


X = np.array(df[['f5', 'f12', 'f23', 'f24', 'f30', 'f3', 'f9', 'f27', 'f18', 'f29', 'f35', 'f15', 'f6', 'f33', 'f11', 'f21', 'f10', 'f0', 'f8', 'f32', 'f17', 'f19']])
Y = np.array(df[['target']])

loocv = model_selection.LeaveOneOut()
knn_loocv = KNeighborsClassifier(n_neighbors=7)
results_loocv = model_selection.cross_val_score(knn_loocv, X, Y, cv=loocv)

m_features.append(21)
accuracy.append(results_loocv.mean()*100.0)


# In[87]:


X = np.array(df[['f5', 'f12', 'f23', 'f24', 'f30', 'f3', 'f9', 'f27', 'f18', 'f29', 'f35', 'f15', 'f6', 'f33', 'f11', 'f21', 'f10', 'f0', 'f8', 'f32', 'f17']])
Y = np.array(df[['target']])

loocv = model_selection.LeaveOneOut()
knn_loocv = KNeighborsClassifier(n_neighbors=7)
results_loocv = model_selection.cross_val_score(knn_loocv, X, Y, cv=loocv)

m_features.append(20)
accuracy.append(results_loocv.mean()*100.0)


# In[88]:


X = np.array(df[['f5', 'f12', 'f23', 'f24', 'f30', 'f3', 'f9', 'f27', 'f18', 'f29', 'f35', 'f15', 'f6', 'f33', 'f11', 'f21', 'f10', 'f0', 'f8', 'f32']])
Y = np.array(df[['target']])

loocv = model_selection.LeaveOneOut()
knn_loocv = KNeighborsClassifier(n_neighbors=7)
results_loocv = model_selection.cross_val_score(knn_loocv, X, Y, cv=loocv)

m_features.append(19)
accuracy.append(results_loocv.mean()*100.0)


# In[89]:


X = np.array(df[['f5', 'f12', 'f23', 'f24', 'f30', 'f3', 'f9', 'f27', 'f18', 'f29', 'f35', 'f15', 'f6', 'f33', 'f11', 'f21', 'f10', 'f0', 'f8']])
Y = np.array(df[['target']])

loocv = model_selection.LeaveOneOut()
knn_loocv = KNeighborsClassifier(n_neighbors=7)
results_loocv = model_selection.cross_val_score(knn_loocv, X, Y, cv=loocv)

m_features.append(18)
accuracy.append(results_loocv.mean()*100.0)


# In[90]:


X = np.array(df[['f5', 'f12', 'f23', 'f24', 'f30', 'f3', 'f9', 'f27', 'f18', 'f29', 'f35', 'f15', 'f6', 'f33', 'f11', 'f21', 'f10', 'f0']])
Y = np.array(df[['target']])

loocv = model_selection.LeaveOneOut()
knn_loocv = KNeighborsClassifier(n_neighbors=7)
results_loocv = model_selection.cross_val_score(knn_loocv, X, Y, cv=loocv)

m_features.append(17)
accuracy.append(results_loocv.mean()*100.0)


# In[93]:


accuracy_df = pd.DataFrame({'FeaturesNumber': m_features, 'Accuracy': accuracy})
accuracy_df


# ##### We have considered the features whose correlation is less than 0.2. Now, the accuracy of the model is highest with number of features 23. m value of 26 is giving the highest accuracy.

# ### Wrapper Method

# In[99]:


feature_selector = SequentialFeatureSelector(knn_loocv,
           k_features=25,
           forward=True,
           verbose=2,
           scoring='roc_auc',
           cv=4)


# In[101]:


X = np.array(df[['f5', 'f12', 'f23', 'f24', 'f30', 'f3', 'f9', 'f27', 'f18', 'f29', 'f35', 'f15', 'f6', 'f33', 'f11', 'f21', 'f10', 'f0', 'f8', 'f32', 'f17', 'f19', 'f25', 'f28', 'f2']])
Y = np.array(df[['target']])

features = feature_selector.fit(np.array(X), Y)


# In[102]:


features


# In[108]:


filtered_features= df.columns[list(features.k_feature_idx_)]
filtered_features


# ### We had considered all the features in our model. It is showing the first 25 features as best model by forward selection. 

# ### Next we are checking what is the accuracy with those features.

# In[109]:


X = np.array(df[filtered_features])
Y = np.array(df[['target']])


# In[110]:


loocv = model_selection.LeaveOneOut()
knn_loocv = KNeighborsClassifier(n_neighbors=7)
results_loocv = model_selection.cross_val_score(knn_loocv, X, Y, cv=loocv)

m_features.append(17)
print ('Accuracy = ', results_loocv.mean()*100.0)


# In[111]:


print ('Accuracy = ', results_loocv.mean()*100.0)


# ## Accuracy of the FInal model is 92.45%.

# In[ ]:




