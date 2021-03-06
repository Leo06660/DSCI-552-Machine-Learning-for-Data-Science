#!/usr/bin/env python
# coding: utf-8

# In[12]:


import csv
import numpy as np
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers


# ## Part (a) [3.5 points]: 
# Find the fattest margin line that separates the points in linsep.txt. Please solve the problem using a Quadratic Programming solver. Report the equation of the line as well as the support vectors.

# In[115]:


linesp_data = []
with open('linsep.txt') as f:
    file = csv.reader(f)
    for line in file:
        linesp_data.append(line)
linesp_data = np.array(linesp_data).astype('float64')


# In[116]:


X = linesp_data[:,:2]
y = linesp_data[:,2]
n_samples, n_features = X.shape


# In[117]:


# Q = y y X^T X
K = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        K[i,j] = np.dot(X[i], X[j])
Q = matrix(np.outer(y, y) * K)


# In[118]:


q = matrix(np.ones((n_samples, 1)) * -1)
A = matrix(y.reshape(1, -1))
b = matrix(np.zeros(1))          
G = matrix(np.eye(n_samples) * -1)
h = matrix(np.zeros(n_samples))


# In[119]:


solution = solvers.qp(Q, q, G, h, A, b)
# find the support vectors
alphas = np.array(solution['x'])
idx = (alphas > 1e-4).flatten()
sv = X[idx]
sv_y = y[idx]
alphas = alphas[idx]


# In[120]:


# print the support vector
print('---alphas---')
print(alphas)
print('-----sv-----')
print(sv)
print('----sv_y----')
print(sv_y)


# In[121]:


# Calculate w (exclude alpha = 0)
w = np.zeros(n_features)
for n in range(len(alphas)):
    w += alphas[n]*sv_y[n]*sv[n]
w


# In[102]:


# Calculate b
b = sv_y[0] - np.dot(sv[0], w)
b


# In[103]:


# plot all data points and mark them respectively
pos_idx = np.where(linesp_data[:,2]==1)
neg_idx = np.where(linesp_data[:,2]==-1)
X_pos = linesp_data[pos_idx]
X_neg = linesp_data[neg_idx]
plt.scatter(X_pos[:,0], X_pos[:,1], marker = 'o', color = 'b', label = 'Positive +1')
plt.scatter(X_neg[:,0], X_neg[:,1], marker = 'x', color = 'r', label = 'Negative -1')
# plot maximum margin separating hyperplane
x_plane = np.linspace(-1,1,100)
y_plane = (-b-w[0]*x_plane)/w[1]
plt.plot(x_plane, y_plane, color='black')
plt.plot(x_plane, (1/w[1])+y_plane, '--', color='black')
plt.plot(x_plane, (-1/w[1])+y_plane, '--', color='black')
# circle the support vecotor(sv)
plt.scatter(sv[:,0], sv[:,1], s=150, facecolors='none', edgecolors='k')
plt.xlim(-0.025, 1.025)
plt.ylim(-0.025, 1.025)
plt.show()


#  

# # Part (b) [3.5 points]: 
# Using a kernel function of your choice along with the same Quadratic Programming solver, find the equation of a curve that separates the points in nonlinsep.txt. Report the kernel function you use as well as the support vectors.

# In[161]:


nonlinesp_data = []
with open('nonlinsep.txt') as f:
    file = csv.reader(f)
    for line in file:
        nonlinesp_data.append(line)
nonlinesp_data = np.array(nonlinesp_data).astype('float64')


# In[162]:


X = nonlinesp_data[:,:2]
y = nonlinesp_data[:,2]
n_samples, n_features = X.shape


# In[163]:


def RBF_kernel_function(x1, x2, gamma=0.01):
    return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)


# In[164]:


# new Q from kernel function(K)
K = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        K[i,j] = RBF_kernel_function(X[i], X[j])
Q = matrix(np.outer(y, y) * K)


# In[165]:


q = matrix(np.ones((n_samples, 1)) * -1)
A = matrix(y.reshape(1, -1))
b = matrix(np.zeros(1))          
G = matrix(np.eye(n_samples) * -1)
h = matrix(np.zeros(n_samples))


# In[166]:


solution = solvers.qp(Q, q, G, h, A, b)
alphas = np.array(solution['x'])
# find the support vectors
ind = (alphas > 1e-4).flatten()
sv = X[ind]
sv_y = y[ind]
alphas = alphas[ind]


# In[167]:


# print the support vector
print('---alphas---')
print(alphas)
print('-----sv-----')
print(sv)
print('----sv_y----')
print(sv_y)


# In[154]:


# plot all data points and mark them respectively
pos_idx = np.where(nonlinesp_data[:,2]==1)
neg_idx = np.where(nonlinesp_data[:,2]==-1)
X_pos = nonlinesp_data[pos_idx]
X_neg = nonlinesp_data[neg_idx]
plt.scatter(X_pos[:,0], X_pos[:,1], marker = 'o', color = 'b', label = 'Positive +1')
plt.scatter(X_neg[:,0], X_neg[:,1], marker = 'x', color = 'r', label = 'Negative -1')
# circle the support vecotor(sv)
plt.scatter(sv[:,0], sv[:,1], s=150, facecolors='none', edgecolors='k')
plt.show()

