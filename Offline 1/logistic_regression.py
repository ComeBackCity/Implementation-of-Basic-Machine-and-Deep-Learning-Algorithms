#!/usr/bin/env python
# coding: utf-8

# In[7]:

# imports
import numpy as np


# In[8]:

# tanh function
def tanh_function(z):
    return (np.exp(z) - np.exp(-z)) / (np.exp(z) + np.exp(-z))


# In[9]:


# loss function
def loss(h, y, size):
    error = np.sum((y - h) ** 2) / size
    return error


# In[10]:


# training function
def train(x, y, early_terminate_threshold=0.0, learning_rate=0.00005, no_of_iterations=5000):
    no_of_data, no_of_features = x.shape
    w = np.random.rand(no_of_features, 1)
    error = 0
    for i in range(no_of_iterations):
        z = np.dot(x, w)
        h = tanh_function(z)
        error = loss(h, y, no_of_data)
        if error < early_terminate_threshold:
            break
        gradient = np.dot(x.T, (y - h) * (1 - h ** 2))
        w += learning_rate * gradient

    print('Error rate in training set = {}'.format(error))
    return w

# In[11]:
