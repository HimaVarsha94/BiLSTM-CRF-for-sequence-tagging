
# coding: utf-8

# In[1]:


import pickle
import numpy


# In[2]:

import sys

lang_pair=sys.argv[1]
with open('../data/duolingo/'+lang_pair+'/procd_'+lang_pair+'_train_allfeats_lowered.pkl', 'rb') as f:
    train_data = pickle.load(f)

with open('../data/duolingo/'+lang_pair+'/procd_'+lang_pair+'_dev_allfeats_lowered.pkl', 'rb') as f:
    dev_data = pickle.load(f)

with open('../data/duolingo/'+lang_pair+'/procd_'+lang_pair+'_train_labels.pkl', 'rb') as f:
    train_labels = pickle.load(f)

with open('../data/duolingo/'+lang_pair+'/procd_'+lang_pair+'_dev_labels.pkl', 'rb') as f:
    dev_labels = pickle.load(f)

with open('../data/duolingo/'+lang_pair+'/procd_'+lang_pair+'_train_seq_feats.pkl', 'rb') as f:
    train_feats = pickle.load(f)

with open('../data/duolingo/'+lang_pair+'/procd_'+lang_pair+'_dev_seq_feats.pkl', 'rb') as f:
    dev_feats = pickle.load(f)


# In[3]:


print(type(train_data))
print(type(train_labels))
print(type(train_feats))


# In[4]:


print(train_data[0])
print(train_labels[0])
# print(train_feats['countries'][0])


# In[6]:


train_order = []
for data in train_data:
    train_order.append(len(data['words']))

dev_order = []
for data in dev_data:
    dev_order.append(len(data['words']))

## get order of indices after sorting
sorted_train_inds = numpy.argsort(train_order)
sorted_dev_inds = numpy.argsort(dev_order)

## now use those indices to sort all data
new_train_data = [train_data[i] for i in sorted_train_inds]
new_dev_data = [dev_data[i] for i in sorted_dev_inds]
new_train_labels = [train_labels[i] for i in sorted_train_inds]
new_dev_labels = [dev_labels[i] for i in sorted_dev_inds]
new_train_feats = [train_feats[i] for i in sorted_train_inds]
new_dev_feats = [dev_feats[i] for i in sorted_dev_inds]

with open('../data/duolingo/'+lang_pair+'/ordered_'+lang_pair+'_train_allfeats_lowered.pkl', 'wb') as f:
    train_data = pickle.dump(train_data,f)

with open('../data/duolingo/'+lang_pair+'/ordered_'+lang_pair+'_dev_allfeats_lowered.pkl', 'wb') as f:
    dev_data = pickle.dump(dev_data,f)

with open('../data/duolingo/'+lang_pair+'/ordered_'+lang_pair+'_train_labels.pkl', 'wb') as f:
    train_labels = pickle.dump(train_labels,f)

with open('../data/duolingo/'+lang_pair+'/ordered_'+lang_pair+'_dev_labels.pkl', 'wb') as f:
    dev_labels = pickle.dump(dev_labels,f)

with open('../data/duolingo/'+lang_pair+'/ordered_'+lang_pair+'_train_seq_feats.pkl', 'wb') as f:
    train_feats = pickle.dump(train_feats,f)

with open('../data/duolingo/'+lang_pair+'/ordered_'+lang_pair+'_dev_seq_feats.pkl', 'wb') as f:
    dev_feats = pickle.dump(dev_feats,f)
