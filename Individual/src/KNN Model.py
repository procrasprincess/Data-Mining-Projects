#!/usr/bin/env python
# coding: utf-8
import pandas as pd
import numpy as np
import operator


train = pd.read_csv('spamtrain.csv')
test = pd.read_csv('spamtest.csv')


def E_Distance(x1, x2, length):
    distance = 0
    for x in range(length):
        distance += np.square(x1[x] - x2[x])
    return np.sqrt(distance)

# making function for defining K-NN model - defined for all 57 features

def knn(trainingSet, testInstance, k):
    distances = {}
    length = testInstance.shape[1]
    for x in range(len(trainingSet)):
        dist = E_Distance(testInstance, trainingSet.iloc[x], length)
        distances[x] = dist[0]
    sortdist = sorted(distances.items(), key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(sortdist[x][0])
    Count = {}  # to get most frequent class of rows
    for x in range(len(neighbors)):
        response = trainingSet.iloc[neighbors[x]][-1]
        if response in Count:
            Count[response] += 1
        else:
            Count[response] = 1
    sortcount = sorted(Count.items(), key=operator.itemgetter(1), reverse=True)
    return (sortcount[0][0], neighbors)

#Predicting for n test-rows and k (parameter if k-NN)
def predict_test_results(n,k):
    test_result = []
    test_short = test.head(n)
    for index, rows in test_short.iterrows(): 
        testlist = [[rows.f1, rows.f2, rows.f3, rows.f4, rows.f5, rows.f6,rows.f7, rows.f8, rows.f9,rows.f10, rows.f11, rows.f12, rows.f13, rows.f14, rows.f15, rows.f16,rows.f17, rows.f18, rows.f19,rows.f20, rows.f21, rows.f22, rows.f23, rows.f24, rows.f25, rows.f26,rows.f27, rows.f28, rows.f29,rows.f30, rows.f31, rows.f32, rows.f33, rows.f34, rows.f35, rows.f36,rows.f37, rows.f38, rows.f39,rows.f40, rows.f41, rows.f42, rows.f43, rows.f44, rows.f45, rows.f46,rows.f47, rows.f48, rows.f49,rows.f50, rows.f51, rows.f52, rows.f53, rows.f54, rows.f55, rows.f56,rows.f57]] 
     
        test1 = pd.DataFrame(testlist) 
    
        result, neigh = knn(train, test1, k)
        test_result.append(result)
    
    obs = []
    observed = test_short.Label
    for i in range(0,len(observed)):
        obs.append(observed[float(i)])
    truecases = 0
    for i in range(0,len(obs)):
        if test_result[i] == obs[i]:
            truecases = truecases + 1
    accuracy = truecases/len(obs)
    
    
    return accuracy, test_result


accuracy, predicted = predict_test_results(10,5)


print(accuracy)
print(predicted)


accuracy




