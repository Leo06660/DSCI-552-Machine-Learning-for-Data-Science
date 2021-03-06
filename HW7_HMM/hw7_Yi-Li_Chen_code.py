#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


class HMM:
    def __init__(self, ann, bnm, pi, O):
        self.A = np.array(ann, np.float)
        self.B = np.array(bnm, np.float)
        self.Pi = np.array(pi, np.float)
        self.O = np.array(O, np.float)
        self.N = self.A.shape[0]
        self.M = self.B.shape[1]
    
    def viterbi(self):
        T = len(self.O)
        I = np.zeros(T, np.float)

        delta = np.zeros((T, self.N), np.float)  
        psi = np.zeros((T, self.N), np.float)

        for i in range(self.N):
            delta[0, i] = self.Pi[i] * self.B[i, int(self.O[0])]
            psi[0, i] = 0

        for t in range(1, T):
            for i in range(self.N):
                delta[t, i] = self.B[i, int(self.O[t])] * np.array([delta[t-1,j] * self.A[j,i]
                    for j in range(self.N)] ).max() 
                psi[t,i] = np.array( [delta[t-1,j] * self.A[j,i] 
                    for j in range(self.N)] ).argmax()

        P_T = delta[T-1, :].max()
        I[T-1] = delta[T-1, :].argmax()
        for t in range(T-2, -1, -1):
            I[t] = psi[t+1, int(I[t+1])]
            
        return I, P_T


# In[3]:


O = [8, 6, 4, 6, 5, 4, 5, 5, 7, 9]
A = [[   0,   1,   0,   0,   0,   0,   0,   0,   0,   0],
     [ 1/2,   0, 1/2,   0,   0,   0,   0,   0,   0,   0],
     [   0, 1/2,   0, 1/2,   0,   0,   0,   0,   0,   0],
     [   0,   0, 1/2,   0, 1/2,   0,   0,   0,   0,   0],
     [   0,   0,   0, 1/2,   0, 1/2,   0,   0,   0,   0],
     [   0,   0,   0,   0, 1/2,   0, 1/2,   0,   0,   0],
     [   0,   0,   0,   0,   0, 1/2,   0, 1/2,   0,   0],
     [   0,   0,   0,   0,   0,   0, 1/2,   0, 1/2,   0],
     [   0,   0,   0,   0,   0,   0,   0, 1/2,   0, 1/2],
     [   0,   0,   0,   0,   0,   0,   0,   0,   1,   0]]
B = [[ 1/2, 1/2,   0,   0,   0,   0,   0,   0,   0,   0],
     [ 1/3, 1/3, 1/3,   0,   0,   0,   0,   0,   0,   0],
     [   0, 1/3, 1/3, 1/3,   0,   0,   0,   0,   0,   0],
     [   0,   0, 1/3, 1/3, 1/3,   0,   0,   0,   0,   0],
     [   0,   0,   0, 1/3, 1/3, 1/3,   0,   0,   0,   0],
     [   0,   0,   0,   0, 1/3, 1/3, 1/3,   0,   0,   0],
     [   0,   0,   0,   0,   0, 1/3, 1/3, 1/3,   0,   0],
     [   0,   0,   0,   0,   0,   0, 1/3, 1/3, 1/3,   0],
     [   0,   0,   0,   0,   0,   0,   0, 1/3, 1/3, 1/3],
     [   0,   0,   0,   0,   0,   0,   0,   0, 1/2, 1/2]]
pi = [1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10, 1/10]


# In[4]:


hmm_res = HMM(A,B,pi,O)
hmm_res.viterbi()

