
# coding: utf-8

# In[126]:


import os
import numpy as np
import pandas as pd
import copy

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import mpld3 as mpl

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate


# In[ ]:


os.chdir('/Users/camilaalmeida/Documents')


# ### Loading data

# In[131]:


# Train and test datasets
train_df = pd.read_csv('/code_challenge_dataset/train.csv', sep=',', header=0, encoding='unicode_escape')
test_df = pd.read_csv('/code_challenge_dataset/test.csv', sep=',', header=0, encoding='unicode_escape')


# In[132]:


train_df.head(3)


# In[133]:


test_df.head(3)


# ### Exploring dataset

# In[134]:


train_df.drop('id',axis=1,inplace=True)
train_df.drop('feature_0',axis=1,inplace=True) # Drop not feature columns

len(train_df)


# In[135]:


train_df.nunique().sum()


# In[136]:


# Check missing values in train data
train_df.isnull().sum()


# In[137]:


train_df.describe()
plt.hist(train_df['target'])
plt.title('Target (S=1 , N=0)')
plt.show()


# ### Preprocessing

# In[138]:


# Replace all missing values as a new category
train_df.fillna('MISSING', inplace=True)


# In[148]:


# Feature engineering
### One-hot-encoding method
features = ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5', 'feature_6', 'feature_7', 'feature_8', 'feature_9', 'feature_10', 'feature_11']

train_objs_num = len(train_df)
dataset = pd.concat(objs=[train_df, test_df], axis=0) # concatenate train and test dfs
dataset = pd.get_dummies(dataset, prefix=features, columns=features, dummy_na=True)

train_final = copy.copy(dataset[:train_objs_num]) # split train and test datasets
test_final = copy.copy(dataset[train_objs_num:])
test_final.drop(['feature_0','id','target'], axis=1, inplace=True)


# In[149]:


test_final.head()


# ### Prediction

# In[150]:


# Logistic Regression
features = train_final.columns[3:]
target = train_final.columns[2:3]

X = train_final[features]
Y = train_final[target]

logmodel = LogisticRegression()
logmodel.fit(X,Y)
predictions = logmodel.predict(test_final)


# ### Model evaluation

# In[127]:


scoring = {'accuracy': 'accuracy', 'log_loss': 'neg_log_loss', 'auc': 'roc_auc'}

modelCV = LogisticRegression()

results = cross_validate(modelCV, X, Y, cv=10, scoring=list(scoring.values()), 
                         return_train_score=False)

print('K-fold cross-validation results:')
for sc in range(len(scoring)):
    print(modelCV.__class__.__name__+" average %s: %.3f (+/-%.3f)" % (list(scoring.keys())[sc], -results['test_%s' % list(scoring.values())[sc]].mean()
                               if list(scoring.values())[sc]=='neg_log_loss' 
                               else results['test_%s' % list(scoring.values())[sc]].mean(), 
                               results['test_%s' % list(scoring.values())[sc]].std()))


# In[142]:


test_final['target'] = logmodel.predict(test_final[features])
test_final['id'] = test_df['id']

results = test_final[['id','target']].astype(int)

results.head()


# In[143]:


results.describe()
plt.hist(results['target'])
plt.title('Target (S=1 , N=0)')
plt.show()


# In[144]:


results.to_csv("results.csv", index=False)

