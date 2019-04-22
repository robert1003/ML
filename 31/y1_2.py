#!/usr/bin/env python
# coding: utf-8

# In[8]:


# specify id
y_id = 1
track_id = 2
server = 4


# In[9]:


# import module
import sys
sys.path.insert(0, '../')
from utils.training_utils import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_absolute_error, make_scorer
from xgboost import XGBRegressor


# In[67]:


# specify parameters
params = {
    'booster': ['gbtree', 'dart'],
    'n_estimators': [50, 100, 500],
    'max_depth': [3, 6],
    'subsample': [0.25, 0.5, 0.75],
    'learning_rate': [0.1, 0.05, 0.01], 
    'tree_method': ['hist', 'auto']
}


# In[43]:


# load datas
test_x, train_x, train_y = load_data(y_id)
print(test_x.shape, train_x.shape, train_y.shape)


# In[47]:


# pick only important data
idx = []
with open('../29/adaboost' + str(y_id) + '_feature.csv', 'r') as f:
    i = 0
    for lines in f:
        importance = float(lines.replace('\n', '').split(',')[y_id])
        if(np.abs(importance) > 1e-9):
            idx.append(i)
        i += 1
train_x = train_x[:, idx]
test_x = test_x[:, idx]
print(train_x.shape)


# In[71]:


# define my own scorer
from sklearn.metrics import make_scorer

def scorer(y, y_pred):
    return -np.sum(np.abs(y - y_pred) / y) / len(y)


# In[13]:


# create GridSearchCV
model = GridSearchCV(estimator=XGBRegressor(verbosity=2, n_jobs=8), 
                     param_grid=params, 
                     scoring=make_scorer(scorer),
                     cv=3,
                     verbose=20,
                     n_jobs=4,
                     return_train_score=True)


# In[14]:


# train
model.fit(train_x, train_y)


# In[15]:


# write files
write_prediction('train_y' + str(y_id) + '_' + str(track_id) + '.txt', 'w', model.predict(train_x).reshape((47500, 1)).astype('str'))
write_prediction('test_y' + str(y_id) + '_' + str(track_id) + '.txt', 'w', model.predict(test_x).reshape((2500, 1)).astype('str'))


# In[16]:


print(err2_calc(model.predict(train_x), train_y, y_id))


# In[1]:


print(model.best_estimator_)


# In[ ]:




