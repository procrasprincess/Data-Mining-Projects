#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
import numpy as np


# In[2]:


train = pd.read_csv('census-incomedata.csv')
test = pd.read_csv('census-incometest.csv')


# In[3]:


print('train dimension :' ,train.shape)
print('test dimension :' ,test.shape)


# In[4]:


## Renaming Column names
train.columns = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','IncomeMoreThan50k']
test.columns = ['age','workclass','fnlwgt','education','education_num','marital_status','occupation','relationship','race','sex','capital_gain','capital_loss','hours_per_week','native_country','IncomeMoreThan50k']


# In[5]:


train.head()


# In[6]:


test.head()


# ### Data Cleaning

# In[7]:


# converting all to lowercase , replacing '-' with '_'
train['workclass'] = train['workclass'].apply(lambda x : x.lower())
train['workclass'] = train['workclass'].apply(lambda x : x.replace(" ", "_"))
train['workclass'] = train['workclass'].apply(lambda x : x.replace("-", "_"))
train['education'] = train['education'].apply(lambda x : x.lower())
train['education'] = train['education'].apply(lambda x : x.replace(" ", "_"))
train['education'] = train['education'].apply(lambda x : x.replace("-", "_"))
train['marital_status'] = train['marital_status'].apply(lambda x : x.lower())
train['marital_status'] = train['marital_status'].apply(lambda x : x.replace(" ", "_"))
train['marital_status'] = train['marital_status'].apply(lambda x : x.replace("-", "_"))
train['occupation'] = train['occupation'].apply(lambda x : x.lower())
train['occupation'] = train['occupation'].apply(lambda x : x.replace(" ", "_"))
train['occupation'] = train['occupation'].apply(lambda x : x.replace("-", "_"))
train['relationship'] = train['relationship'].apply(lambda x : x.lower())
train['relationship'] = train['relationship'].apply(lambda x : x.replace(" ", "_"))
train['relationship'] = train['relationship'].apply(lambda x : x.replace("-", "_"))
train['race'] = train['race'].apply(lambda x : x.lower())
train['race'] = train['race'].apply(lambda x : x.replace(" ", "_"))
train['race'] = train['race'].apply(lambda x : x.replace("-", "_"))
train['sex'] = train['sex'].apply(lambda x : x.lower())
train['native_country'] = train['native_country'].apply(lambda x : x.lower())
train['native_country'] = train['native_country'].apply(lambda x : x.replace(" ", "_"))
train['native_country'] = train['native_country'].apply(lambda x : x.replace("-", "_"))


# In[8]:


# Same data cleaning for test data
test['workclass'] = test['workclass'].apply(lambda x : x.lower())
test['workclass'] = test['workclass'].apply(lambda x : x.replace(" ", "_"))
test['workclass'] = test['workclass'].apply(lambda x : x.replace("-", "_"))
test['education'] = test['education'].apply(lambda x : x.lower())
test['education'] = test['education'].apply(lambda x : x.replace(" ", "_"))
test['education'] = test['education'].apply(lambda x : x.replace("-", "_"))
test['marital_status'] = test['marital_status'].apply(lambda x : x.lower())
test['marital_status'] = test['marital_status'].apply(lambda x : x.replace(" ", "_"))
test['marital_status'] = test['marital_status'].apply(lambda x : x.replace("-", "_"))
test['occupation'] = test['occupation'].apply(lambda x : x.lower())
test['occupation'] = test['occupation'].apply(lambda x : x.replace(" ", "_"))
test['occupation'] = test['occupation'].apply(lambda x : x.replace("-", "_"))
test['relationship'] = test['relationship'].apply(lambda x : x.lower())
test['relationship'] = test['relationship'].apply(lambda x : x.replace(" ", "_"))
test['relationship'] = test['relationship'].apply(lambda x : x.replace("-", "_"))
test['race'] = test['race'].apply(lambda x : x.lower())
test['race'] = test['race'].apply(lambda x : x.replace(" ", "_"))
test['race'] = test['race'].apply(lambda x : x.replace("-", "_"))
test['sex'] = test['sex'].apply(lambda x : x.lower())
test['native_country'] = test['native_country'].apply(lambda x : x.lower())
test['native_country'] = test['native_country'].apply(lambda x : x.replace(" ", "_"))
test['native_country'] = test['native_country'].apply(lambda x : x.replace("-", "_"))


# ### Preparing data for Modelling

# In[9]:


## Changing dpendent varaible to binary flag , 1 indicates income more than 50k , 0 otherwise
train['IncomeMoreThan50k'] = train['IncomeMoreThan50k'].apply(lambda x : x.replace(" <=50K", "0"))
train['IncomeMoreThan50k'] = train['IncomeMoreThan50k'].apply(lambda x : x.replace(" >50K", "1"))
test['IncomeMoreThan50k'] = test['IncomeMoreThan50k'].apply(lambda x : x.replace(" <=50K.", "0"))
test['IncomeMoreThan50k'] = test['IncomeMoreThan50k'].apply(lambda x : x.replace(" >50K.", "1"))


# In[10]:


## Checking for imbalanced
from collections import Counter
Counter(train['IncomeMoreThan50k'])


# * there are 12434 people having less than 50k income and 3846 otherwise
# * the data is quite imbalance

# In[11]:


y_train = train['IncomeMoreThan50k'].astype('int64').values


# In[12]:


y_test = test['IncomeMoreThan50k'].astype('int64').values


# In[13]:


train.columns


# In[14]:


## Encoding Numerival variables

print('fnlwgt')
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
normalizer.fit(train['fnlwgt'].values.reshape(1,-1))

X_train_fnlwgt = normalizer.transform(train['fnlwgt'].values.reshape(1,-1)).transpose()
X_test_fnlwgt = normalizer.transform(test['fnlwgt'].values.reshape(1,-1)).transpose()

print("After vectorizations")
print(X_train_fnlwgt.shape, y_train.shape)
print(X_test_fnlwgt.shape, y_test.shape)
print("="*100)

print('education_num')
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
normalizer.fit(train['education_num'].values.reshape(1,-1))

X_train_education_num = normalizer.transform(train['education_num'].values.reshape(1,-1)).transpose()
X_test_education_num = normalizer.transform(test['education_num'].values.reshape(1,-1)).transpose()

print("After vectorizations")
print(X_train_education_num.shape, y_train.shape)
print(X_test_education_num.shape, y_test.shape)
print("="*100)

print('age')
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
normalizer.fit(train['age'].values.reshape(1,-1))

X_train_age = normalizer.transform(train['age'].values.reshape(1,-1)).transpose()
X_test_age = normalizer.transform(test['age'].values.reshape(1,-1)).transpose()

print("After vectorizations")
print(X_train_age.shape, y_train.shape)
print(X_test_age.shape, y_test.shape)
print("="*100)

print('capital_gain')
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
normalizer.fit(train['capital_gain'].values.reshape(1,-1))

X_train_capital_gain = normalizer.transform(train['capital_gain'].values.reshape(1,-1)).transpose()
X_test_capital_gain = normalizer.transform(test['capital_gain'].values.reshape(1,-1)).transpose()

print("After vectorizations")
print(X_train_capital_gain.shape, y_train.shape)
print(X_test_capital_gain.shape, y_test.shape)
print("="*100)

print('capital_loss')
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
normalizer.fit(train['capital_loss'].values.reshape(1,-1))

X_train_capital_loss = normalizer.transform(train['capital_loss'].values.reshape(1,-1)).transpose()
X_test_capital_loss = normalizer.transform(test['capital_loss'].values.reshape(1,-1)).transpose()

print("After vectorizations")
print(X_train_capital_loss.shape, y_train.shape)
print(X_test_capital_loss.shape, y_test.shape)
print("="*100)

print('hours_per_week')
from sklearn.preprocessing import Normalizer
normalizer = Normalizer()
normalizer.fit(train['hours_per_week'].values.reshape(1,-1))

X_train_hours_per_week = normalizer.transform(train['hours_per_week'].values.reshape(1,-1)).transpose()
X_test_hours_per_week = normalizer.transform(test['hours_per_week'].values.reshape(1,-1)).transpose()

print("After vectorizations")
print(X_train_hours_per_week.shape, y_train.shape)
print(X_test_hours_per_week.shape, y_test.shape)
print("="*100)


# In[15]:


train.columns


# In[16]:


## Encoding Categorical variables
from sklearn.feature_extraction.text import CountVectorizer
print('workclass')
vectorizer = CountVectorizer()
vectorizer.fit(train['workclass'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_workclass = vectorizer.transform(train['workclass'].values)
X_test_workclass = vectorizer.transform(test['workclass'].values)

print("After vectorizations")
print(X_train_workclass.shape, y_train.shape)
print(X_test_workclass.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)

print('education')
vectorizer = CountVectorizer()
vectorizer.fit(train['education'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_education = vectorizer.transform(train['education'].values)
X_test_education = vectorizer.transform(test['education'].values)

print("After vectorizations")
print(X_train_education.shape, y_train.shape)
print(X_test_education.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)

print('marital_status')
vectorizer = CountVectorizer()
vectorizer.fit(train['marital_status'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_marital_status = vectorizer.transform(train['marital_status'].values)
X_test_marital_status = vectorizer.transform(test['marital_status'].values)

print("After vectorizations")
print(X_train_marital_status.shape, y_train.shape)
print(X_test_marital_status.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)

print('occupation')
vectorizer = CountVectorizer()
vectorizer.fit(train['occupation'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_occupation = vectorizer.transform(train['occupation'].values)
X_test_occupation = vectorizer.transform(test['occupation'].values)

print("After vectorizations")
print(X_train_occupation.shape, y_train.shape)
print(X_test_occupation.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)

print('relationship')
vectorizer = CountVectorizer()
vectorizer.fit(train['relationship'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_relationship = vectorizer.transform(train['relationship'].values)
X_test_relationship = vectorizer.transform(test['relationship'].values)

print("After vectorizations")
print(X_train_relationship.shape, y_train.shape)
print(X_test_relationship.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)

print('race')
vectorizer = CountVectorizer()
vectorizer.fit(train['race'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_race = vectorizer.transform(train['race'].values)
X_test_race = vectorizer.transform(test['race'].values)

print("After vectorizations")
print(X_train_race.shape, y_train.shape)
print(X_test_race.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)

print('sex')
vectorizer = CountVectorizer()
vectorizer.fit(train['sex'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_sex = vectorizer.transform(train['sex'].values)
X_test_sex = vectorizer.transform(test['sex'].values)

print("After vectorizations")
print(X_train_sex.shape, y_train.shape)
print(X_test_sex.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)

print('native_country')
vectorizer = CountVectorizer()
vectorizer.fit(train['native_country'].values) # fit has to happen only on train data

# we use the fitted CountVectorizer to convert the text to vector
X_train_native_country = vectorizer.transform(train['native_country'].values)
X_test_native_country = vectorizer.transform(test['native_country'].values)

print("After vectorizations")
print(X_train_native_country.shape, y_train.shape)
print(X_test_native_country.shape, y_test.shape)
print(vectorizer.get_feature_names())
print("="*100)


# In[17]:


## CONCATINATING THE FEATURES
from scipy.sparse import hstack
X_tr = hstack((X_train_fnlwgt,X_train_education_num,X_train_age,X_train_capital_gain,X_train_capital_loss,
                X_train_hours_per_week,X_train_workclass,X_train_education,X_train_marital_status,
                X_train_occupation,X_train_relationship,X_train_race,X_train_sex,X_train_native_country)).tocsr()



X_te = hstack((X_test_fnlwgt,X_test_education_num,X_test_age,X_test_capital_gain,X_test_capital_loss,
                X_test_hours_per_week,X_test_workclass,X_test_education,X_test_marital_status,
                X_test_occupation,X_test_relationship,X_test_race,X_test_sex,X_test_native_country)).tocsr()

print("Final Data matrix")
print(X_tr.shape, y_train.shape)
print(X_te.shape, y_test.shape)
print("="*100)


# ### Building Decision Tree

# In[18]:


from sklearn.metrics import confusion_matrix 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import accuracy_score 
from sklearn.metrics import classification_report


# In[ ]:


import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

DT = DecisionTreeClassifier(class_weight="balanced") ## balancing the class weights
tree_para = {'max_depth':[1,5,10,50,100,500,1000],'min_samples_split':[5,10,100,500]}
clf = GridSearchCV(DT, tree_para, cv=10, scoring='accuracy')
clf.fit(X_tr, y_train)

#best = clf.best_estimator_

df_gridsearch = pd.DataFrame(clf.cv_results_)


# In[21]:


import seaborn as sns
print("Train Accuracy scores")
max_scores = df_gridsearch.groupby(['param_max_depth', 
                                    'param_min_samples_split']).max()
max_scores = max_scores.unstack()[['mean_train_score']]
sns.heatmap(max_scores.mean_train_score, annot=True, fmt='.4g');


# In[22]:


print("CV Accuracy Scores")
max_scores = df_gridsearch.groupby(['param_max_depth', 
                                    'param_min_samples_split']).max()
max_scores = max_scores.unstack()[['mean_test_score']]
sns.heatmap(max_scores.mean_test_score, annot=True, fmt='.4g');


# * We select max_depth = 5 and min_sample split = 5 , as the accuracy difference between the train and cv is less here
# * It is advisable to select hyperparameter in such a way that the accuracy difference is less , because high difference indicated the model is overfitting

# In[27]:


## Training the model the model
clf = DecisionTreeClassifier(max_depth=5, min_samples_split=5,class_weight="balanced")
clf.fit(X_tr, y_train)


# In[28]:


## Prediction
y_pred = clf.predict(X_te)


# In[29]:


print("Confusion Matrix:")
confusion_matrix(y_test, y_pred)


# In[30]:


print ('Train accuracy : ')
print ("Test Accuracy : ",accuracy_score(y_test,y_pred)*100)


# In[31]:


from sklearn.metrics import precision_score

print("precision_score : ", 
    precision_score(y_test, y_pred)) 


# In[32]:


from sklearn.metrics import recall_score

print("recall_score : ", 
    recall_score(y_test, y_pred)) 


# In[ ]:




