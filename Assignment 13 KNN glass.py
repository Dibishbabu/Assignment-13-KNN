#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


# In[2]:


df = pd.read_csv("D:\KNN/glass.csv")
df


# In[3]:


X= df.iloc[:,:9]
Y= df.iloc[:,9]


# # Standardizing the data

# In[4]:


def get_standardized_data(data):
    df_norm = (data-data.min())/(data.max()-data.min())
    return(df_norm)


# In[5]:


X=get_standardized_data(X)
X


# In[6]:


df["Type"].value_counts()


# In[7]:


#Since the class 5 has just 4 data points hence we keep the n_splits as 3


# In[8]:


kfold=KFold(n_splits=9)


# In[9]:


model = KNeighborsClassifier(n_neighbors=5)
model.fit(X,Y)
cvs=cross_val_score(model,X,Y,cv=kfold)
print(cvs.mean())


# In[10]:


noofneighbours = [2*i+1 for i in range(0,10)]
cvsl =[]
for neighbours in noofneighbours:
    model = KNeighborsClassifier(n_neighbors=neighbours)
    model.fit(X,Y)
    cvs=cross_val_score(model,X,Y,cv=9)
    print("no of neighbours: "+str(neighbours)+"     Average Score: "+str(cvs.mean()))
    cvsl.append(cvs.mean())


# In[11]:


plt.bar(noofneighbours,cvsl)
plt.xticks(noofneighbours)
plt.ylim(0.0,1.4)


# In[12]:


model = KNeighborsClassifier(n_neighbors=3)
model.fit(X,Y)
cvs=cross_val_score(model,X,Y,cv=9)
cvs.mean()


# # Checking model on 2 random datapoints

# In[13]:


X= df.iloc[212:214,:9]
Y= df.iloc[212:214,9]


# In[14]:


model.predict(X)


# In[15]:


Y


# Even though 1 gives higher accuracy , we choose 3 as optimal number beacuse just considering 1 nearest neighbour is inadequate to reach a conclusion

# # Hence optimum number of neighbours is 3 with accuracy 0.654 for KNN

# In[ ]:




