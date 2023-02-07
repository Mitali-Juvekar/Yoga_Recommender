#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


df = pd.read_csv("C:/Users/mital/Documents/Python proj/yoga.csv")


# In[3]:


df.head()


# In[4]:


df.shape


# In[6]:


df['Benefits']


# In[10]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


# In[9]:


tfidf = TfidfVectorizer(stop_words="english")
df['Benefits'] = df['Benefits'].fillna("")
tfidf_matrix = tfidf.fit_transform(df['Benefits'])


# In[11]:


cosine_sin = linear_kernel(tfidf_matrix, tfidf_matrix)


# In[12]:


df['Asana']


# In[18]:


indices = pd.Series(df.index,index = df["Asana"]).drop_duplicates()
indices


# In[19]:


indices['SIDEWAYS VIEWING']


# In[24]:


def recommendations(asana, cosine_sin = cosine_sin):
    index = indices[asana]
    sin_scores = enumerate(cosine_sin[index])
    sin_scores = sorted(sin_scores, key=lambda x: x[1], reverse=True)
    sin_scores = sin_scores[1:4]
    sin_index = [i[0] for i in sin_scores]
    print(df['Asana'].iloc[sin_index])
        
# recommendations('SIDEWAYS VIEWING')


# In[25]:


recommendations('SIDEWAYS VIEWING')


# In[27]:


recommendations(input("Symptom"))


# In[ ]:





# In[ ]:




