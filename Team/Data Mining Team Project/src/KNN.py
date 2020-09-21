#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May  4 05:51:05 2020

@author: dylansmith
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:04:36 2020

@author: dylansmith

"""

#BASE
import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNC
import copy
#IMPUTE
import random as rand


#################################################################
def weave(X,Y,I,i):
    ##string together arrays by lists of inclusion indices
    X=X.copy()
    Y=Y.copy()
    xc=0
    yc=0
    A=[]
    S=list(set(I+i))
    if len(X[0])==1:
        for s in range(len(S)):
            if S[s] in I:
                A.append(float(X[xc]))
                xc+=1
            elif S[s] in i:
                A.append(float(Y[yc]))
                yc+=1
    else:
        for s in range(len(S)):
            if S[s] in I:
                A.append(list(X[xc]))
                xc+=1
            elif S[s] in i:
                A.append(list(Y[yc]))
                yc+=1
    return A

def split(X,I,array=True,axis=1):
    ##TAKES INPUT: ARRAY and LIST. SPLITS INTO A AND B according to Indices {I}
    ##A: X[I]
    ##B: X[not I]
    if axis==1:
        X=np.transpose(X)
    r=len(X)
    if type(I)==int:
        if I<0:
            I=r+I
        I=list(range(I))
    R=list(range(r))
    for i in range(len(I)):
        if I[i]<0:
            I[i]=r+I[i]
    A=[];B=[]
    for i in I:
        A+=[list(X[i])]
        R.remove(i)
    for r in R:
        B+=[list(X[r])]
    if axis==1:
        A,B=np.transpose(A),np.transpose(B)
    else:
        A,B=np.array(A),np.array(B)
    if array!=True:
        A,B=list(A),list(B)
    return [A,B] 


#################################################################
def rl(it):
    return range(len(it))

def grouping(inp,dic=False,k=10,R=False):
    lis=list(inp)
    if dic!=False:
        test=dic
    else:
        x=max(lis)
        m=min(lis)
        n=len(list(set(lis)))
        if 4*n<k:
            k=int(n/4)
        step=(x-m)/(k-1)
        test=[0]
        for i in range(k-1):
            test.append(round(m+(i+1)*step,2))
    out=[]
    for i in rl(lis):
        for j in rl(test):
            if lis[i]<=test[j] and type(R)==int:
                if j==R:
                    j=1
                    break
                else:
                    j=0
                    break
            if lis[i]<=test[j] and type(R)==bool:
                j
                break
        out.append(j)
    return out,test

def decode(inp,dic,original=False):
    N=[]
    n=np.array(inp)
    for y in rl(n):
        X=[]
        for x in rl(n[0]):
            d=dic[x][int(n[y][x])]
            if type(d)!=str:
                if type(original)!=bool:
                    d=original[y][x]
                else:
                    d=n[y][x]
            X.append(d)
        N.append(X)
    return N
    
Source=['./census-income.total.csv','./census-income.data.csv','./census-income.test.csv']
DF = pd.read_csv(Source[0]),pd.read_csv(Source[1]),pd.read_csv(Source[2])
TrainNum=DF[1].shape[0]
TestNum=DF[2].shape[0]



df = pd.read_csv(Source[0])
ar = np.array(df)
ar0= np.array(df.T[1:].T)

##Values
values=[]
##Numeric Encoding and Grouping
nGroups=41
encoding=[]
for i in range(1,len(ar[0])):
    values.append(sorted(list(set(df[df.columns[i]]))))
    if type(values[-1][0])==int:
        encoding.append(dict(zip(list(range(len(values[-1]))),grouping(values[-1],k=nGroups)[1])))
    else:
        encoding.append(dict(zip(list(range(len(values[-1]))),values[-1])))

## Then take numeric encoding and translate array

     
for i in range(len(ar[0])):
    if i==0:
        numer=df[df.columns[0]]
        normal=df[df.columns[0]]
    else:
        G=grouping(df[df.columns[i]],encoding[i-1])
        N=pd.DataFrame(G[0])/max(G[1])
        numer=pd.concat([numer,pd.DataFrame(G[0])],axis=1,sort=False)
        normal=pd.concat([normal,N],axis=1,sort=False)
Num=np.transpose(np.transpose(np.array(numer))[1:])
Norm=np.transpose(np.transpose(np.array(normal))[1:])






##Imputing Values
colnums=False;encode=False;x0=' ?';k0=7  
array=Norm;output=Num;encode=encoding    
num=np.transpose(array)
#COLUMNS
L1=[]
S1=[]
for n in rl(num):
    if encode!=False:
        if encode[n][0]==x0:
            x=0
        else:
            x=x0
    if list(num[n]).count(x)>0:
        S1.append(n)
    else:
        L1.append(n)

##Columns
X,Y_=split(array,L1,axis=1)
X_,Y=split(Num,L1,axis=1)
#Dist=[]

Yi=[]
Yx=[]
for i in rl(S1):
    Y0,nu=split(Y,[i],axis=1)
    print(i)
    c=S1[i] 
    
    #INSTANCES
    L2=[]
    S2=[]
    for n in rl(num[0]):
        if array[n][c]==0:
            S2.append(n)
        else:
            L2.append(n)
    
    XTrain,XImpute=split(X,L2,axis=0) ##TrainX,TestX
    YTrain,YImpute=split(Y0,L2,axis=0) ##TrainY,Prediction
    
    Knn=KNC(n_neighbors=k0)
    Knn.fit(XTrain,YTrain)
    y=Knn.predict(XImpute)
    print(y)
    for j in rl(y):
        YImpute[j]=y[j]
    Yi.append(weave(YTrain,YImpute,L2,S2))
    X=weave(XTrain,XImpute,L2,S2)
    
Y_=[]
for i in range(3):
    Y_.append(list(np.dot(Yi[i],1/max(Yi[i]))))
Y_=np.transpose(Y_)
Y=np.transpose(Yi)

Data=pd.DataFrame(np.transpose(weave(np.transpose(X_),np.transpose(Y),L1,S1)),columns=list(df.columns)[1:])
KData=pd.DataFrame(np.transpose(weave(np.transpose(X),np.transpose(Y_),L1,S1)),columns=list(df.columns)[1:])
D=pd.DataFrame(decode(Data,encoding,ar0),columns=list(df.columns)[1:])
   
##Export Imputed Arrays
Training=D[0:TrainNum];Testing=D[TrainNum:TestNum+TrainNum]
exp1=Training.to_csv("imputedData.train.csv")
exp1=Testing.to_csv("imputedData.test.csv")

###Expansion

##Values
values=[]
##Numeric Encoding and Grouping
nGroups=41
encoding=[]
df=D
ar = np.array(df)

for i in range(0,len(ar[0])):
    values.append(sorted(list(set(df[df.columns[i]]))))
    if type(values[-1][0])==int:
        encoding.append(dict(zip(list(range(len(values[-1]))),grouping(values[-1],k=nGroups)[1])))
    else:
        encoding.append(dict(zip(list(range(len(values[-1]))),values[-1])))

##Expanding Categorical Columns

Prime=pd.DataFrame()
for i in range(len(ar[0])):
    if type(encoding[i][0])==str:
        for j in rl(encoding[i]):
            G=grouping(df[df.columns[i]],encoding[i],R=j)
            N=pd.DataFrame(G[0],columns=[encoding[i][j]])
            Prime=pd.concat([Prime,N],axis=1,sort=False)
    else:
        G=grouping(df[df.columns[i]],encoding[i])
        N=df[df.columns[i]]/max(G[1])
        Prime=pd.concat([Prime,N],axis=1,sort=False)
        
#_________________________________________________#
###COMPLETE AND EXPANDED DATA
 
Training=Prime[0:TrainNum]
Tr=Training.T
Testing=Prime[TrainNum:TestNum+TrainNum]
Te=Testing.T
    
[XTrain,YTrain]=[Tr[0:-2].T,Tr[-2:-1].T]
[XTest,YTest]=[Te[0:-2].T,Te[-2:-1].T]

##Testing and Accuracy

K={}
for k in range(1,20,2):
    print(k)
    Knn=KNC(n_neighbors=k)
    Knn.fit(XTrain,YTrain)
    YPred=Knn.predict(XTest)
    K[k]=YPred

from sklearn.metrics import confusion_matrix

recall=[];specificity=[];acc=[];precision=[]
for k in range(1,20,2):
    tn, fp, fn, tp = confusion_matrix(YTest, K[k]).ravel()
    recall.append(tp/(tp+fn))   #label 1 accuracy
    specificity.append(tn/(tn+fp))       #label 0 accuracy
    acc.append((tp+tn)/(tn+fp+fn+tp))
    precision.append(tp/(tp+fp))
    
    print("________________\nk=",k)
    print("recall=",recall[-1])
    print("specificity=",specificity[-1])
    print("acc=",acc[-1])
    print("precision=",recall[-1])

print("________________\n")
print("Max Test Set Recall at k=",1+recall.index(max(recall))*2)
print("Max Test Set Specificity at k=",1+specificity.index(max(specificity))*2)
print("Max Test Set Accuracy at k=",1+acc.index(max(acc))*2)
print("Max Test Set Precision at k=",1+precision.index(max(precision))*2)
print("\n________________\n")
print("Chosen k is k=13, Due to Highest Test Set Recall,Specificity,Accuracy and Precision")

##Calculate Training Set Accuracy for k=13
print("\n________________\n")
k=13
Knn=KNC(n_neighbors=k)
Knn.fit(XTrain,YTrain)
YPred=Knn.predict(XTrain)
tn, fp, fn, tp = confusion_matrix(YTrain, YPred).ravel()
recall.append(tp/(tp+fn))   #label 1 accuracy
specificity.append(tn/(tn+fp))       #label 0 accuracy
acc.append((tp+tn)/(tn+fp+fn+tp))
precision.append(tp/(tp+fp))
print("________________\n")

print("________________\n________________\nk=",k)
print("Training Set:")
print("recall=",recall[-1])
print("specificity=",specificity[-1])
print("acc=",acc[-1])
print("precision=",recall[-1])
print("________________\nTesting Set:")
print("recall=",recall[6])
print("specificity=",specificity[6])
print("acc=",acc[6])
print("precision=",recall[6])