#!/usr/bin/env python
# coding: utf-8

# In[18]:


import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


# In[19]:


p=r"C:\Users\zbook 17 g3\Downloads\sms.csv"


# In[20]:


df=pd.read_csv(p,sep='\t',names=['Status','Message'])


# In[21]:


df.head()


# In[22]:


len(df[df.Status=="ham"])


# In[23]:


len(df[df.Status=="spam"])


# In[24]:


df.loc[df["Status"]=="ham","Status",]=1


# In[25]:


df.loc[df["Status"]=="spam","Status",]=0


# In[26]:


df.head()


# In[27]:


df.tail()


# In[28]:


df_x=df["Message"]
df_y=df["Status"]


# In[29]:


x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=4)


# In[30]:


x_train.head()


# In[31]:


cv1 = TfidfVectorizer(min_df=1,stop_words='english')


# In[32]:


x_traincv=cv1.fit_transform(x_train)
a=x_traincv.toarray()
a[0]


# In[33]:


x_testcv=cv1.transform(x_test)
x_testcv.toarray()


# In[35]:


mnb = MultinomialNB()


# In[36]:


y_train=y_train.astype('int')
y_train


# In[37]:


mnb.fit(x_traincv,y_train)


# In[38]:


predictions=mnb.predict(x_testcv)


# In[39]:


predictions


# In[40]:


y_test


# In[41]:


a=np.array(y_test)


# In[43]:


a


# In[44]:




import matplotlib.pyplot as plt
aa=1115-1068

labels = ["true","false"]
sizes = [1115,aa]

plt.figure(figsize=(7, 7))
plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140)

plt.title("classes percentage")

plt.axis('equal')
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




