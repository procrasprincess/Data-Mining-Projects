import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB

train_data = pd.read_csv("imputedData.train.csv")
test_data = pd.read_csv("imputedData.test.csv")

le = LabelEncoder()

train_data['Work-class']= le.fit_transform(train_data['Work-class'])
train_data['Education']= le.fit_transform(train_data['Education'])
train_data['Marital-status']= le.fit_transform(train_data['Marital-status'])
train_data['Occupation']= le.fit_transform(train_data['Occupation'])
train_data['Relationship']= le.fit_transform(train_data['Relationship'])
train_data['Race']= le.fit_transform(train_data['Race'])
train_data['Sex']= le.fit_transform(train_data['Sex'])
train_data['Native-country']= le.fit_transform(train_data['Native-country'])
train_data[' <=50K']= le.fit_transform(train_data[' <=50K'])

test_data['Work-class']= le.fit_transform(test_data['Work-class'])
test_data['Education']= le.fit_transform(test_data['Education'])
test_data['Marital-status']= le.fit_transform(test_data['Marital-status'])
test_data['Occupation']= le.fit_transform(test_data['Occupation'])
test_data['Relationship']= le.fit_transform(test_data['Relationship'])
test_data['Race']= le.fit_transform(test_data['Race'])
test_data['Sex']= le.fit_transform(test_data['Sex'])
test_data['Native-country']= le.fit_transform(test_data['Native-country'])
test_data[' <=50K']= le.fit_transform(test_data[' <=50K'])

# onehotencoder = OneHotEncoder()
# data = onehotencoder.fit_transform(x_train).toarray()
# data

x_train = train_data.iloc[:,:-1]
y_train = train_data.iloc[:,-1]

x_test = test_data.iloc[:,:-1]
y_test = test_data.iloc[:,-1]

model = GaussianNB()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print("TN:", matrix[0][0])
print("FP:", matrix[0][1])
print("FN:", matrix[1][0])
print("TP:", matrix[1][1])

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Precision:", precision_score(y_test, y_pred))
print("Recall:", recall_score(y_test, y_pred))
print("F1:", f1_score(y_test, y_pred))
print("Specificity:", matrix[0][0] / (matrix[0][0] + matrix[0][1]))
