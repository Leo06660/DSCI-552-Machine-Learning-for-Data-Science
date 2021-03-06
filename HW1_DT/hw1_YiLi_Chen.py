#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Group16 YiLi Chen


# In[2]:


import pandas as pd
from math import log 


# # PART1

# ## (a)

# In[3]:


def get_orginal_entropy(df, target):
    orginal_e = 0
    # The probs of enjoy and not enjoy
    probs = df.groupby(target).size().div(len(df))
    for p in probs:
        orginal_e += -p*log(p, 2)
    return orginal_e


# In[4]:


def get_feature_entropy(col_name, df, target): #Get entropy of each branch
    feature_e = 0
    # The probs of each categorized group 
    probs = df.groupby(col_name).size().div(len(df))
    # iterate all groups and sum the entropy
    for index_value, p_value in list(probs.items()):
        # calculate each group probs
        group_df = df[df[col_name] == index_value]
        group_probs = group_df.groupby(target).size().div(len(group_df))
        group_entropy=0 # remove the previous group's value
        # Sum H(C|when x=xi)
        for p in group_probs:
            group_entropy += -p*log(p,2)
        # Sum H(target|col_name)
        feature_e += p_value * group_entropy
    return feature_e


# In[5]:


def get_max_info_gain(df, target):
    d = {}
    orginal_e = get_orginal_entropy(df, target)
    all_col_name = list(df.columns)
    all_col_name.pop(-1)
    for col_name in all_col_name:
        d[col_name] = orginal_e - get_feature_entropy(col_name, df, target)
    return max(d, key=d.get)


# In[6]:


def split_df(df, best_feature):
    splitDict = {}
    for k in df[best_feature].unique():
        splitDict[k] = df[df[best_feature]==k].drop(columns=best_feature)
    return splitDict


# In[7]:


def create_decision_tree(df, target):
    # if the original_e less than threshold
    if get_orginal_entropy(df, target) < 0.1:
        return df.iloc[:,-1].value_counts().index[0]
    # if left only one type of feature, then return
    if len(df.columns) == 1:
        last_col_name = df.columns[0]
        return df[last_col_name].value_counts().index[0]
    # find max info gain of feature
    fname = get_max_info_gain(df, target) 
    node = {fname: {}}    
    
    #build tree 
    df2 = split_df(df, fname)
    for t,d in df2.items():#zip(df2.keys(),df2.values()):
        node[fname][t] = create_decision_tree(d, d.iloc[:,-1])  
    return node


# ## (b)

# In[8]:


df = pd.read_csv('dt_data.txt', sep=",", header=None)
df = df.drop([0])
df.columns = ['Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer', 'Enjoy']
df = df.reset_index()
df = df.drop(columns=['index'])
for i in range(22):
    df['Occupied'][i] = df['Occupied'][i].lstrip('0123456789: ')
    df['Enjoy'][i] = df['Enjoy'][i].rstrip(';').lstrip(' ')
    for j in range(6):
        df.iloc[:,j][i] = df.iloc[:,j][i].lstrip(' ')


# In[9]:


tree_model = create_decision_tree(df, df['Enjoy'])
tree_model


# ## (c)

# In[10]:


def tree_predict(tree_model, feature_names, data):
    key = list(tree_model.keys())[0]
    tree_model = tree_model[key]

    pred=None
    for k in tree_model:
        # find the corresponding attribute
        if data[key][0] == k:
            # judge is it dict or not
            if isinstance(tree_model[k], dict):
                # if yes, keep finding in the next dict inside
                pred = tree_predict(tree_model[k], feature_names, data)
            else:
                # if no, return the label
                pred = tree_model[k]
    return pred


# In[11]:


data = pd.DataFrame(data = [['Moderate','Cheap','Loud','City-Center','No','No']],columns=['Occupied', 'Price', 'Music', 'Location', 'VIP', 'Favorite Beer'])
data


# In[12]:


tree_predict(tree_model, list(data.columns), data)

