#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!/usr/bin/env python
# coding: utf-8

# In[6]:


import numpy as np


# In[7]:


def sigmoid(Z):
    A = 1.0/(1.0 + np.exp(-Z))
    cache = Z
    return A, cache


# In[12]:


def sigmoid_prime(dA, cache):
    Z = cache
    s = 1.0/(1.0 + np.exp(-Z))
    return dA * s*(1-s)


# In[9]:


def relu(Z):
    cache = Z
    A = np.maximum(0, Z)
    return A, cache


# In[10]:


def relu_prime(dA, cache):
    Z = cache
    dZ = np.array(dA, copy=True)
    dZ[Z<=0] = 0
    return dZ


# In[ ]:





# In[ ]:




