#!/usr/bin/env python
# coding: utf-8

# # THE SPARKS FOUNDATION
# ## TASK 4 : COMPUTER VISION AND IoT

# ## Color Identification in Images: Implement an image color detector which identifies all the colors in an image or video

# ### By SUPARNA SARKAR

# **Importing the libraries**

# In[1]:


import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
import cv2

get_ipython().run_line_magic('matplotlib', 'inline')


# **Reading an image**

# In[2]:


def read_n_get_image(file_path):
    image = cv2.imread(file_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image


# In[3]:


image = read_n_get_image("1.jpg")
plt.imshow(image)


# **Resizing the image**

# In[4]:


# Resizing of image is done so that KMeans does not take too much time to return the prominent colors
image = cv2.resize(image, (500, 500))


# In[5]:


# Show the image after resizing
plt.imshow(image)


# **Creating the matrix of features for K-Means Algorithm**

# In[6]:


print(type(image))
print(image)
print(image.shape)


# In[7]:


# x is the matrix of features to be supplied to the KMeans Algorithm
x = image.reshape(image.shape[0]*image.shape[1],3)


# In[8]:


print(x)


# **Building the KMeans Clustering Model to find out the top five prominent colors in the model**

# In[9]:


kmeans = KMeans(n_clusters=5)


# In[10]:


kmeans.fit(x)


# In[11]:


prominent_colors = kmeans.cluster_centers_
prominent_colors = prominent_colors.astype(int)
print(prominent_colors)


# In[12]:


plt.imshow([prominent_colors])
plt.axis('off')
plt.title("Prominent Colors")


# **Getting the ordering of prominency among the colors**

# In[13]:


# Get the number of pixels participating in each cluster
intensity_list = Counter(kmeans.labels_)
print(intensity_list)


# In[14]:


keys = list(intensity_list.keys())
values = list(intensity_list.values())
print(keys)
print(values)


# In[15]:


# Show the prominency levels on a bar graph
bar_graph = plt.bar(keys,values)
for x in range(0, len(prominent_colors)):
    bar_graph[x].set_color(prominent_colors[keys[x]].astype(float)/255.0)
plt.show(bar_graph)

