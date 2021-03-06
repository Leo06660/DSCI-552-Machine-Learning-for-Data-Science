#!/usr/bin/env python
# coding: utf-8

# In[1]:


import csv
import numpy as np
from math import *
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal


# In[2]:


data = []
with open('clusters.txt') as file:
    reader = csv.reader(file)
    for line in reader:
        data.append([float(line[0]),float(line[1])])


# # k-means

# In[384]:


def initial_centroids(data, k):
    num_samples, dim = np.array(data).shape
    centroids = np.zeros((k, dim))
    idx = np.random.randint(len(data), size=3)
    for i in range(k):
        centroids[i] = data[idx[i]]
    return centroids


# In[385]:


def min_distance(p1, p2):
    return sqrt((p1[0]-p2[0])**2+(p1[1]-p2[1])**2)


# In[386]:


def update2centroids(datapoint, centroids):
    k_num_distance = []
    for c in centroids:
        d = min_distance(c, datapoint)
        k_num_distance.append(d)
    return np.argmin(k_num_distance)


# In[387]:


def clasify2cluster(data, k, centroids, cluster):
    cluster=[[] for i in range(k)]
    for p in data:
        c_idx = update2centroids(p, centroids)
        cluster[c_idx].append(p)
    # find new centroids from the clusters
    num_samples, dim = np.array(data).shape
    new_centroids = np.zeros((k, dim))
    for i in range(k):
        new_x = np.array(cluster[i])[:,0].mean()
        new_y = np.array(cluster[i])[:,1].mean()
        new_centroids[i] = [new_x,new_y]
        difference = new_centroids - centroids
    return (difference, new_centroids, cluster)


# In[388]:


def kmeans(data, k):
    mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
    centroids_dict = {'cluster1':[],'cluster2':[],'cluster3':[]}
    cluster=[[] for i in range(k)]
    centroids = initial_centroids(data, k)
    centroids_dict['cluster1'].append(tuple(centroids[0]))
    centroids_dict['cluster2'].append(tuple(centroids[1]))
    centroids_dict['cluster3'].append(tuple(centroids[2]))
    difference, new_centroids, cluster = clasify2cluster(data, k, centroids, cluster)
    while np.any(difference != 0):
        cluster=[[] for i in range(k)]
        difference, new_centroids, cluster = clasify2cluster(data, k, new_centroids, cluster)
#         print(new_centroids)
        for i in range(3):
            plt.plot(np.array(cluster[i])[:,0],np.array(cluster[i])[:,1],mark[i])
        plt.show()
        centroids_dict['cluster1'].append(tuple(new_centroids[0]))
        centroids_dict['cluster2'].append(tuple(new_centroids[1]))
        centroids_dict['cluster3'].append(tuple(new_centroids[2]))
#         for key_index in range(3):
#             centroids_dict[list(centroids_dict.keys())[i]].append(new_centroids[i])
    return (difference, new_centroids, cluster, centroids_dict)


# In[ ]:


difference, new_centroids, cluster, centroids_dict = kmeans(data, 3)


# In[390]:


difference, new_centroids, cluster, centroids_dict = kmeans(data, 3)


# In[391]:


df = pd.DataFrame(centroids_dict)
df


# In[ ]:





# # GMM

# In[19]:


def initial_ric(data, k):
    initial_ric = np.zeros((len(data), k))
    random_int = np.random.randint(low=1,high=10000,size=(len(data), k))
    for row in range(len(random_int)):
        initial_ric[row] = random_int[row]/sum(random_int[row])
    return initial_ric


# In[6]:


def caluculate_new_mu(k, ric, data):
    new_mu = np.zeros((k, 2))
    # calculate the mu of each cluster
    for c_index in range(k):
        # loop all datapoints and their corresponding ric
        for r, p in zip(ric[:,c_index], np.array(data)):
            new_mu[c_index] += r*p
        new_mu[c_index] = new_mu[c_index] / sum(ric[:,c_index])
    return new_mu


# In[7]:


def caluculate_new_sigma(k, ric, data, mu):
    new_sigma = np.array([[[0, 0], [0, 0]] for i in range(0,k)], dtype='float64')
    for c_index in range(k):
        # loop all datapoints and their corresponding ric
        for r, p in zip(ric[:,c_index], np.array(data)):
            new_sigma[c_index] += r*np.outer(p-mu[c_index],p-mu[c_index])
        new_sigma[c_index] = new_sigma[c_index] / sum(ric[:,c_index])
    return new_sigma


# In[8]:


def caluculate_new_pi(k, ric, data):
    new_pi = np.zeros(k)
    for idx in range(k):
        new_pi[idx] = sum(ric[:,idx]) / len(data)
    return new_pi


# In[9]:


def update_ric(k, data, mu, sigma, pi):
    pdfs = np.zeros(((len(data), k)))
    for i in range(k):
        pdfs[:, i] = pi[i] * multivariate_normal.pdf(data, mu[i], sigma[i])
    ric = pdfs / pdfs.sum(axis=1).reshape(-1, 1)
    return ric


# In[20]:


def GMM(data, k):
    param_dict = {'mu':[],'covariance':[],'pi':[]}
    diff = float('Inf')
    ric = initial_ric(data, k)
    while np.any(abs(diff) > 0.005):
        mu = caluculate_new_mu(k, ric, data)
        sigma = caluculate_new_sigma(k, ric, data, mu)
        pi = caluculate_new_pi(k, ric, data)
        new_ric = update_ric(k, data, mu, sigma, pi)
        # store all updated parameters for each iteration of GMM
        param_dict['mu'].append(mu)
        param_dict['covariance'].append(sigma)
        param_dict['pi'].append(pi)
        diff = new_ric - ric
        ric = new_ric
        # classify and store the data points into list
        cluster=[[] for i in range(k)]
        for idx in range(len(ric)):
            c_idx = np.argmax(ric[idx])
            cluster[c_idx].append(data[idx])
    # plot the cluster result    
    for i in range(3):
        mark = ['or', 'ob', 'og', 'ok', '^r', '+r', 'sr', 'dr', '<r', 'pr']
        plt.plot(np.array(cluster[i])[:,0],np.array(cluster[i])[:,1],mark[i])
    plt.show()
    return (mu, sigma, pi, ric, param_dict)


# In[21]:


mu, sigma, pi, ric, param_dict = GMM(data, 3)


# In[29]:


print('mu')
print(param_dict['mu'][-1])
print('\ncovariance')
print(param_dict['covariance'][-1])
print('\npi')
print(param_dict['pi'][-1])


# In[25]:


df2 = pd.DataFrame(param_dict)
df2

