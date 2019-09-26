#!/usr/bin/env python
# coding: utf-8

# In[32]:


import numpy as np
import pandas as pd
from platform import python_version

print(python_version())


# In[65]:


class LDA():
    def __init__(self):
        self.w1 = None
        self.w0 = None
    
    def fit(self, X_train, y_train):
        data = X_train
        data['quality'] = y_train
        
        u0 = data.loc[data['quality']==0].iloc[:, :len(data.columns)-1]
        u1 = data.loc[data['quality']==1].iloc[:, :len(data.columns)-1]
        
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
        
        cov0 = np.cov(np.transpose(u0))
        cov1 = np.cov(np.transpose(u1))
        covariance_matrix = cov1 + cov0
        
        division = second_class_prob/first_class_prob
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        class_two = 0.5*float(np.matmul(np.matmul(second_average,inv_covariance_matrix),np.transpose(second_average)))
        class_one = 0.5*float(np.matmul(np.matmul(first_average,inv_covariance_matrix),np.transpose(first_average)))
        
        self.w0 = np.log(division) - class_two + class_one
        self.w1 = np.matmul(inv_covariance_matrix, (np.transpose(second_average)-np.transpose(first_average)))
        
    def predict(self, X_test):
        X_test = X_test.to_numpy()
        return [1 if np.dot(x,self.w1) + self.w0 > 0 else 0 for x in X_test]


# In[76]:


data = pd.read_csv("winequality-red.csv", sep=';').astype('float32')
X_train = data.iloc[:, :len(data.columns)-1]
y_train = data.iloc[:, len(data.columns)-1:].values
y_train = [0 if i < 6 else 1 for i in y_train]
data['quality'] = y_train

def k_fold(data, n):
    results = []
    fold_length = int(len(data)/5)
    for i in range(n):
        start = fold_length*i
        end = start + fold_length
        temp_data = data
        
        temp_data_test = temp_data.iloc[start:end]
        
        X_test = temp_data_test.iloc[:, :len(temp_data_test.columns)-1]
        y_test = temp_data_test.iloc[:, len(temp_data_test.columns)-1:].values
        
        top = temp_data.iloc[0:start]
        bottom = temp_data.iloc[end:len(temp_data)]
        
        temp_data_train = pd.concat([top,bottom])
        
        X_train = temp_data_test.iloc[:, :len(temp_data_train.columns)-1]
        y_train = temp_data_test.iloc[:, len(temp_data_train.columns)-1:].values
        
        lda = LDA()
        lda.fit(X_train, y_train)
        y_pred = lda.predict(X_test)
        
        correct = 0.0
        for pred,true in zip(y_pred, y_test):
            if pred == true:
                correct += 1 
        results.append(correct/len(y_train))
    return results
        
results = k_fold(data, 5)
avg = np.mean(results)
std = np.std(results)
print(avg)
print(std)


# In[53]:


data = pd.read_csv("winequality-red.csv", sep=';').astype('float32')


# In[4]:


X_train = data.iloc[:, :len(data.columns)-1]
y_train = data.iloc[:, len(data.columns)-1:].values
y_train = [0 if i < 6 else 1 for i in y_train]

data.insert(11, "result", y_train)
data = data.iloc[:, :len(data.columns)-1]


# In[38]:


u0 = data
u0 = data.loc[data['result']==0].iloc[:, :len(u0.columns)-1]


# In[39]:


u1 = data
u1 = data.loc[data['result']==1].iloc[:, :len(u1.columns)-1]


# In[40]:


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





# In[41]:


# covariance_matrix_one = (1.0 / (len(y_train)-1)) * (u0 - u0.mean(axis=0)).T.dot(u0 - u0.mean(axis=0))
# covariance_matrix_two = (1.0 / (len(y_train)-1)) * (u1 - u1.mean(axis=0)).T.dot(u1 - u1.mean(axis=0))
# covariance_matrix = covariance_matrix_one + covariance_matrix_two
# covariance_matrix

cov0 = np.cov(np.transpose(u0))
cov1 = np.cov(np.transpose(u1))
covariance_matrix = cov1 + cov0


# In[ ]:





# In[48]:


division = second_class_prob/first_class_prob
inv_covariance_matrix = np.linalg.inv(covariance_matrix)
class_two = 0.5*float(np.matmul(np.matmul(second_average,inv_covariance_matrix),np.transpose(second_average)))
class_one = 0.5*float(np.matmul(np.matmul(first_average,inv_covariance_matrix),np.transpose(first_average)))


# In[49]:


w0 = np.log(division) - class_two + class_one
w1 = np.matmul(inv_covariance_matrix, (np.transpose(second_average)-np.transpose(first_average)))


# In[50]:


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


# In[51]:


cov0 = np.cov(np.transpose(u0))
cov1 = np.cov(np.transpose(u1))


# In[ ]:





# In[ ]:




