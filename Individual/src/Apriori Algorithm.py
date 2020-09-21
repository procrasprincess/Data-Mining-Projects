#!/usr/bin/env python
# coding: utf-8

# # Question 1 Apriori Algorithm

# In[127]:


A=np.array([[0,0,1,0,1,0],[0,1,1,1,0,1],[1,0,0,0,1,0],[1,1,1,0,0,0],[0,0,0,1,0,0],[1,0,0,1,0,1],[0,0,1,1,1,1],[1,0,1,0,1,0],[1,0,0,1,0,0],[0,1,1,0,0,1]])
#level 1 transactions features vs transactions(a,b,c,d,e,f are features)
d1={'a':A[:,0],'b':A[:,1],'c':A[:,2],'d':A[:,3],'e':A[:,4],'f':A[:,5]}
d1


# In[119]:


C1=(np.sum(A,axis=0))
#Level 1 Candidate sets count of number of transactions for each feature
C1


# In[23]:


Support_C1=C1/10


# In[24]:


#Level1 Support 
Support_C1


# In[25]:


#if support threshold is 0.3,then number of features that satisfy more than support
L1_num=np.sum(Support_C1>0.3)
L1_num


# In[120]:


#Level 1 support as a fraction of total number of transactions that satisfy support>0.3
L1_support=Support_C1[Support_C1>0.3]
L1_support


# In[126]:


#for level 2 the transactions with more than one transaction
B=A[np.sum(A,axis=1)>1]
B


# In[72]:


d2={'ab':(B[B[:,0]+B[:,1]==2]),'ac':(B[B[:,0]+B[:,2]==2]),'ad':(B[B[:,0]+B[:,3]==2]),'ae':(B[B[:,0]+B[:,4]==2]),'af':(B[B[:,0]+B[:,5]==2]),'bc':(B[B[:,1]+B[:,2]==2]),'bd':(B[B[:,1]+B[:,3]==2]),'be':(B[B[:,1]+B[:,4]==2]),'bf':(B[B[:,1]+B[:,5]==2]),'cd':(B[B[:,2]+B[:,3]==2]),'ce':(B[B[:,2]+B[:,4]==2]),'cf':(B[B[:,2]+B[:,5]==2]),'de':(B[B[:,3]+B[:,4]==2]),'df':(B[B[:,3]+B[:,5]==2]),'ef':(B[B[:,4]+B[:,5]==2])}


# In[93]:


#Level 2 transactions and their features
for i,j in enumerate(d2):
    if np.sum(d2[j])==0:
        del d2[j]
    else:
        print(j)
        print(d2[j])
        


# In[106]:


#Use support of atleast two transactions for level 2
for i,j in enumerate(d2):
    if len(d2[j])>1:
        print(j)
        print(d2[j])


# In[115]:


d3={'abc':(B[B[:,0]+B[:,1]+B[:,2]==3]),'abd':(B[B[:,0]+B[:,1]+B[:,3]==3]),'abe':(B[B[:,0]+B[:,1]+B[:,3]==3]),'abf':B[B[:,0]+B[:,1]+B[:,5]==3],'acd':B[B[:,0]+B[:,2]+B[:,3]==3],'ace':B[B[:,0]+B[:,2]+B[:,4]==3],'acf':B[B[:,0]+B[:,2]+B[:,5]==3],'ade':B[B[:,0]+B[:,4]+B[:,5]==3],'adf':B[B[:,0]+B[:,3]+B[:,5]==3],'aef':B[B[:,0]+B[:,4]+B[:,5]==3],'bcd':B[B[:,1]+B[:,2]+B[:,4]==3],'bce':B[B[:,1]+B[:,2]+B[:,4]==3],'bcf':B[B[:,1]+B[:,3]+B[:,5]==3],'bde':B[B[:,1]+B[:,3]+B[:,4]==3],'bef':B[B[:,1]+B[:,4]+B[:,5]==3],'cde':B[B[:,2]+B[:,3]+B[:,4]==3],'cdf':B[B[:,2]+B[:,3]+B[:,5]==3],'cef':B[B[:,2]+B[:,4]+B[:,5]==3],'def':B[B[:,3]+B[:,4]+B[:,5]==3]}


# In[117]:


#Consider support for atleast one transaction, below are ouputs for third level.
for i,j in enumerate(d3):
    if len(d3[j])>=1:
        print(j)
        print(d3[j])


# #1(a) So from first three levels, maximal frequent sets are df,cf,ce. 
#       df:
#       Support with df = 3/10 = 0.3  since d and f both have 3 transactions together out of 10 transactions
#       Confidance with df = 3/4 =0.75 since d with f is in three transactions out of 4  transactions in which 'f' is there
#       cf:
#       Support with cf = 3/10 = 0.3  since d and f both have 3 transactions together out of 10 transactions
#       Confidance with cf = 3/4 =0.75 since d with f is in three transactions out of 4  transactions in which 'f' is there
#        ce:
#       Support with ce = 3/10 = 0.3  since d and f both have 3 transactions together out of 10 transactions
#       Confidance with ce = 3/3 =1 since c with e is in three transactions out of 3  transactions in which 'e' is there
#       Please refer python dictionary d1 below for better understanding

# In[128]:


d1


# In[136]:


get_ipython().system('pip install pyfpgrowth')


# # Question 2 FP growth

# In[137]:


import pandas as pd
import numpy as np
import pyfpgrowth


# In[142]:


transactions = [['a', 'b', 'e'],
                ['a', 'b','c','d'],
                ['a', 'c','d'],
                ['a', 'c', 'e'],
                ['b','c','f'],
                ['a'],
                ['a','b','c'],
                ['b','d','e'],
                ['a','c'],['a','b','d','e']]


# In[143]:


patterns = pyfpgrowth.find_frequent_patterns(transactions, 2)


# In[144]:


#Conditional pattern
rules = pyfpgrowth.generate_association_rules(patterns, 0.7)


# In[145]:


#2(a) FP tree frequent Patters
patterns


# In[146]:


#2(b) d's Conditional Pattern base considering probability cutoff of 0.7
rules


# In[147]:


patterns = pyfpgrowth.find_frequent_patterns(rules, 2)


# In[148]:


#2(b) Frequent patterns based on d's conditional FP tree
patterns


# # Qustion3 GSP Algorithm

# In[3]:


d1={'1':[0,0,0,1,1,1,0],'2':[1,1,0,1,1,0,0],'3':[1,1,1,1,0,1,1],'4':[1,0,1,0,0,0,1],'5':[0,1,1,0,1,1,1]}
d1


# In[9]:


import numpy as np
A=np.array([[0,1,1,1,0],[0,1,1,0,1],[0,0,1,1,1],[1,1,1,0,0],[1,1,0,0,1],[1,0,1,0,1],[0,0,1,1,1]])
C1=(np.sum(A,axis=0))
#Level 1 Candidate sets count of number of transactions for each feature
C1


# In[10]:


#for level 2 the transactions with more than one transaction
B=A[np.sum(A,axis=1)>1]
B


# In[11]:


d2={'12':(B[B[:,0]+B[:,1]==2]),'13':(B[B[:,0]+B[:,2]==2]),'14':(B[B[:,0]+B[:,3]==2]),'15':(B[B[:,0]+B[:,4]==2]),'23':(B[B[:,1]+B[:,2]==2]),'24':(B[B[:,1]+B[:,3]==2]),'25':(B[B[:,1]+B[:,4]==2]),'34':(B[B[:,2]+B[:,3]==2]),'35':(B[B[:,2]+B[:,4]==2]),'45':(B[B[:,3]+B[:,4]==2])}


# In[14]:


#Level 2 transactions and their features
for i,j in enumerate(d2):
    if np.sum(d2[j])==0:
        d2.pop(j)
    else:
        print(j)
        print(d2[j])


# In[15]:


#Use support of atleast two transactions for level 2
for i,j in enumerate(d2):
    if len(d2[j])>1:
        print(j)
        print(d2[j])


# In[17]:


d3={'123':(B[B[:,0]+B[:,1]+B[:,2]==3]),'124':(B[B[:,0]+B[:,1]+B[:,3]==3]),'125':(B[B[:,0]+B[:,1]+B[:,3]==3]),'134':B[B[:,0]+B[:,2]+B[:,3]==3],'135':B[B[:,0]+B[:,2]+B[:,4]==3],'145':B[B[:,0]+B[:,3]+B[:,4]==3],'234':B[B[:,1]+B[:,2]+B[:,4]==3],'235':B[B[:,1]+B[:,2]+B[:,4]==3],'245':B[B[:,1]+B[:,3]+B[:,4]==3],'345':B[B[:,2]+B[:,3]+B[:,4]==3]}


# In[18]:


#Consider support for atleast one transaction, below are ouputs for third level.
for i,j in enumerate(d3):
    if len(d3[j])>=1:
        print(j)
        print(d3[j])


# In[22]:


d4={'1234':[],'1235':[],'1245':[],'1345':[],'2345':[]}


# # Question 4 DTW distance

# In[23]:


def dtw(s, t):

    n, m = len(s), len(t)

    dtw_matrix = np.zeros((n+1, m+1))

    for i in range(n+1):

        for j in range(m+1):

            dtw_matrix[i, j] = np.inf

    dtw_matrix[0, 0] = 0

    

    for i in range(1, n+1):

        for j in range(1, m+1):

            cost = abs(s[i-1] - t[j-1])

            # take last min from a square box

            last_min = np.min([dtw_matrix[i-1, j], dtw_matrix[i, j-1], dtw_matrix[i-1, j-1]])

            dtw_matrix[i, j] = cost + last_min

    return dtw_matrix


# In[24]:


s=[39,44,43,39,46,38,39,43]
t=[37,44,41,44,39,39,39,40]


# In[25]:


dtw(s,t)


# In[28]:


from fastdtw import fastdtw

from scipy.spatial.distance import euclidean



x = np.array([1, 2, 3, 3, 7])

y = np.array([1, 2, 2, 2, 2, 2, 2, 4])



distance, path = fastdtw(x, y, dist=euclidean)



print(distance)

print(path)


# In[27]:


get_ipython().system('pip install fastdtw')


# In[ ]:




