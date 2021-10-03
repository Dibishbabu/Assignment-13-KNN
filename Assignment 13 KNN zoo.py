#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt


# In[2]:


df=pd.read_csv("D:/KNN/Zoo.csv")
df


# In[3]:


X= df.iloc[:,1:17]
Y = df.iloc[:,17]
X


# In[4]:


def get_standardized_val(data):
    df_norm = (data-data.min())/(data.max()-data.min())
    return(df_norm)


# In[5]:


X=get_standardized_val(X)
X


# # Building the model

# In[6]:


model = KNeighborsClassifier(n_neighbors=5)
model.fit(X,Y)


# In[7]:


df["type"].value_counts()


# In[8]:


#Since the class 5 has just 4 data points hence we keep the n_splits as 3


# In[9]:


kfold = KFold(n_splits=3)


# In[10]:


cvs=cross_val_score(model,X,Y,cv=kfold)


# In[11]:


print(cvs.mean())


# # Checking model on 2 random datapoints

# In[12]:


predict_val = df.iloc[99:101,1:17]
predict_val


# In[13]:


model.predict(predict_val)


# In[14]:


df.iloc[99:101,17]


# # Finding optimal number of neighbours

# In[15]:


optimal_neighbours = [2*i+1 for i in range(0,10)]
cvsl =[]
for size in optimal_neighbours:
    model=KNeighborsClassifier(n_neighbors=size)
    model.fit(X,Y)
    cvs=cross_val_score(model,X,Y,cv=3)
    print("no of neighbours: "+str(size)+"     Average Score: "+str(cvs.mean()))
    cvsl.append(cvs.mean())


# In[16]:


plt.bar(optimal_neighbours,cvsl)
plt.xticks(optimal_neighbours)
plt.ylim(0.6,1.2)


# In[19]:


model=KNeighborsClassifier(n_neighbors=3)
model.fit(X,Y)
cvs=cross_val_score(model,X,Y,cv=kfold)
cvs.mean()


# In[18]:


model.predict(predict_val)


# Even though 1 gives higher accuracy , we choose 3 as optimal number beacuse just considering 1 nearest neighbour is inadequate to reach a conclusion

# # Hence optimum number of neighbours is 3 with accuracy 0.9096 for KNN
