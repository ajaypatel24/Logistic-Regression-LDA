#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[5]:


class LDL(self):
    
    def fit(X_train, y_train):
        for 
        


# In[28]:


data = pd.read_csv("datasets/winequality-red.csv", sep=';').astype('float32')


# In[29]:


X_train = data.iloc[:, :len(data.columns)-1]
y_train = data.iloc[:, len(data.columns)-1:].values
y_train = [0 if i < 6 else 1 for i in y_train]

data.insert(11, "result", y_train)
data = data.iloc[:, :len(data.columns)-1]


# In[30]:


u0 = data
u0 = data.loc[data['result']==0].iloc[:, :len(u0.columns)-1]


# In[31]:


u1 = data
u1 = data.loc[data['result']==1].iloc[:, :len(u1.columns)-1]


# In[32]:


first_class_count = float(len(u0))
second_class_count = float(len(u1))

first_class_prob = first_class_count/len(y_train)
second_class_prob = second_class_count/len(y_train)

first_average = []
second_average = []
for column in u0:
    column = u0[column].values
    first_average.append(np.mean(column))

for column in u1:
    column = u1[column].values
    second_average.append(np.mean(column))

first_average = np.transpose(np.expand_dims(first_average, axis=1))
second_average = np.transpose(np.expand_dims(second_average, axis=1))


# In[ ]:





# In[33]:


# covariance_matrix_one = (1.0 / (len(y_train)-1)) * (u0 - u0.mean(axis=0)).T.dot(u0 - u0.mean(axis=0))
# covariance_matrix_two = (1.0 / (len(y_train)-1)) * (u1 - u1.mean(axis=0)).T.dot(u1 - u1.mean(axis=0))
# covariance_matrix = covariance_matrix_one + covariance_matrix_two
# covariance_matrix

cov0 = np.cov(np.transpose(u0))
cov1 = np.cov(np.transpose(u1))
covariance_matrix = cov1 + cov0


# In[34]:


covariance_matrix


# In[35]:


division = second_class_prob/first_class_prob
inv_covariance_matrix = np.linalg.inv(covariance_matrix)
class_two = 0.5*float(np.matmul(np.matmul(second_average,inv_covariance_matrix),np.transpose(second_average)))
class_one = 0.5*float(np.matmul(np.matmul(first_average,inv_covariance_matrix),np.transpose(first_average)))


# In[37]:


w0 = np.log(division) - class_two + class_one
w1 = np.matmul(inv_covariance_matrix, (np.transpose(second_average)-np.transpose(first_average)))


# In[42]:


correct = 0.0
for x,y in zip(X_train.values, y_train):
    value = np.dot(x,w1) + w0
    if value > 0:
        value = 1
    else:
        value = 0
    if y == value:
        correct += 1
    
print(correct/len(y_train))


# In[393]:


cov0 = np.cov(np.transpose(u0))
cov1 = np.cov(np.transpose(u1))


# In[396]:


(cov0 + cov1)


# In[ ]:




