#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import WordPunctTokenizer


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


df = pd.read_csv('ugd thesis.csv')


# In[4]:


df.head()


# In[5]:


df.isnull().sum()


# In[6]:


udf = df[['user_id', 'business_id', 'stars', 'text']]


# In[7]:


import string


# In[8]:


import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')
stop = []
#to import stopwords in english only, 22 languages are available
for word in stopwords.words('english'):
    s = [char for char in word if char not in string.punctuation]
    stop.append(''.join(s))
print(stopwords.words('english'))


# In[9]:


def clean_text(uarg):
    """
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in uarg if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return " ".join([word for word in nopunc.split() if word.lower() not in stop])


# In[10]:


udf['text'] = udf['text'].dropna().apply(clean_text)


# In[11]:


def getclass(stars):
    score = int(stars)
    if score > 3:
        return 1
    else:
        return 0

def delmore(fulltext):
    if fulltext[-1] == "e" and fulltext[-2] == "r" and fulltext[-3] == "o" and fulltext[-4] == "M":
        return fulltext[0:-8]
    else:
        return fulltext

df["sentiment"] = udf["stars"].apply(getclass)
df["id"] = range(0,len(udf))
df["text"] = udf["text"].apply(delmore)
df_senti = df[["user_id","business_id","text", "sentiment"]]
df_senti.tail()


# In[12]:


#can split original dataset into input column x and output column y
#The training set is the set of data we analyse (train on) to design the rules in the model. A training set is also known as the in-sample data or training data.
#The validation set is a set of data that we did not use when training our model that we use to assess how well these rules perform on new data.
#The test set is a set of data we did not use to train our model or use in the validation set to inform our choice of parameters/input features.
#Data which we use to design our models (Training set)
#Data which we use to refine our models (Validation set)
#Data which we use to test our models (Testing set)
X_train, X_test, y_train, y_test = train_test_split(df_senti['text'], df_senti['sentiment'], test_size=0.33, random_state=42)


# In[13]:


userid_df = df_senti[['user_id','text']]
business_df = df_senti[['business_id', 'text']]


# In[14]:


print(business_df.columns)


# In[15]:


userid_df.head()


# In[16]:


userid_df[userid_df['user_id']=='d693419']['text']


# In[17]:


business_df.head()


# In[18]:


import os
#The os. path module is a very extensively used module that is handy when processing files from different places in the system. 
#It is used for different purposes such as for merging, normalizing and retrieving path names in python


# In[19]:


print(business_df.columns)


# In[20]:


userid_df = userid_df.dropna().groupby('user_id').agg({'text': ' '.join})
business_df = business_df.dropna().groupby('business_id').agg({'text': ' '.join})


# In[21]:


print(business_df.columns)


# In[22]:


userid_df.head()


# In[23]:


business_df.head()


# In[24]:


userid_df.loc['d10035061']['text']
#loc[] method is a method that takes only index labels and returns row or dataframe if the 
#index label exists in the caller data frame.


# In[25]:


from sklearn.feature_extraction.text import CountVectorizer


# In[26]:


userid_vectorizer = TfidfVectorizer(tokenizer = WordPunctTokenizer().tokenize, max_features=None)
userid_vectors = userid_vectorizer.fit_transform(userid_df['text'])
userid_vectors.shape


# In[32]:


userid_vectors


# In[33]:


businessid_vectorizer = TfidfVectorizer(tokenizer = WordPunctTokenizer().tokenize, max_features=None)
businessid_vectors = businessid_vectorizer.fit_transform(business_df['text'])
businessid_vectors.shape


# In[34]:


userid_rating_matrix = pd.pivot_table(df_senti, values='sentiment', index=['user_id'], columns=['business_id'])
userid_rating_matrix.shape


# In[35]:


userid_rating_matrix.head() #here sparsity is high i.e. a user might have gone to few of the restaurants not all


# In[36]:


P = pd.DataFrame(userid_vectors.toarray(), index=userid_df.index, columns=userid_vectorizer.get_feature_names())
Q = pd.DataFrame(businessid_vectors.toarray(), index=business_df.index, columns=businessid_vectorizer.get_feature_names())


# In[37]:


Q.head()


# In[39]:


def matrix_factorization(R, P, Q, steps=25, gamma=0.001,lamda=0.02):
    for step in range(steps):
        for i in R.index:
            for j in R.columns:
                if R.loc[i,j]>0:
                    eij=R.loc[i,j]-np.dot(P.loc[i],Q.loc[j])
                    P.loc[i]=P.loc[i]+gamma*(eij*Q.loc[j]-lamda*P.loc[i])
                    Q.loc[j]=Q.loc[j]+gamma*(eij*P.loc[i]-lamda*Q.loc[j])
        e=0
        for i in R.index:
            for j in R.columns:
                if R.loc[i,j]>0:
                    e= e + pow(R.loc[i,j]-np.dot(P.loc[i],Q.loc[j]),2)+lamda*(pow(np.linalg.norm(P.loc[i]),2)+pow(np.linalg.norm(Q.loc[j]),2))
        if e<0.001:
            break
        
    return P,Q


# In[40]:


get_ipython().run_cell_magic('time', '', 'P, Q = matrix_factorization(userid_rating_matrix, P, Q, steps=25, gamma=0.001,lamda=0.02)')


# In[41]:


Q.head()


# In[42]:


Q.iloc[0].sort_values(ascending=False).head(10)


# In[43]:


business_df.head()


# In[44]:


words = "i want to have dinner with beautiful views and pasta and live music"
test_df= pd.DataFrame([words], columns=['text'])
test_df['text'] = test_df['text'].apply(clean_text)
test_vectors = userid_vectorizer.transform(test_df['text'])
test_v_df = pd.DataFrame(test_vectors.toarray(), index=test_df.index, columns=userid_vectorizer.get_feature_names())

predictItemRating=pd.DataFrame(np.dot(test_v_df.loc[0],Q.T),index=Q.index,columns=['Rating'])
topRecommendations=pd.DataFrame.sort_values(predictItemRating,['Rating'],ascending=[0])[:7]

for i in topRecommendations.index:
    print(df[df['business_id']==i]['business_id'].iloc[0])
    print(df[df['business_id']==i]['text'].iloc[0])
    print(str(df[df['business_id']==i]['stars'].iloc[0]))
    print('')

