#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd


# In[29]:


df = pd.read_csv('Sample.csv')


# In[30]:


df.shape


# In[31]:


df


# In[32]:


df.isnull().sum()


# In[33]:


df['Income'].fillna(df['Income'].median(), inplace = True)


# In[34]:


df


# In[35]:


df.describe()


# In[36]:


pd.DataFrame(df['Age'].describe(percentiles = (1,0.99,0.9,0.75,0.6,0.5,0.4,0.3,0.25,0.1)))


# In[37]:


import matplotlib.pyplot as plt


# In[38]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[39]:


plt.boxplot(df['Age'])


# In[40]:


age_df = pd.DataFrame(df['Age'])
median = df['Age'].median()

q3 = age_df.quantile(q=0.75)
q1 = age_df.quantile(q=0.25)

iqr = q3-q1

iqr_ll = int(q1- (1.5*iqr))
iqr_ul = int(q3+ (1.5*iqr))
print(iqr_ul)


df.loc[df['Age']>iqr_ul,'Age'] = int(age_df.quantile(q=0.99))

df.loc[df['Age']<iqr_ll,'Age'] = int(age_df.quantile(q=0.01))



# In[41]:


max(df['Age'])


# In[42]:


df


# In[44]:


plt.scatter(x=df["Income"],y=df["Expense"])


# In[46]:


plt.scatter(x=df["Age"],y=df["Income"])


# In[51]:


import seaborn as sns
corr_matrix = df.corr().round(2)
sns.heatmap(corr_matrix,annot = True)


# In[53]:


# Feature Engineering 

# Feature/ Predictors /Independent Variables are same 

# 1. Play around with data to achieve goals
# 2. Prepare the proper input datset compatible with ML algorithm requirements
# 3. Improve the performance of ML models. 


# In[62]:


# 1. Normalization of Data/ Scaling the Data
# 2. We will bring all the data to same scale 
# 3. Ex: Min max scaler, using this it will scale all the values between 0 and 1 
# 4. This will be helpful when giving this data as input to ML model as it will be easy for it to understand and compute

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df)
scaled_data = pd.DataFrame(scaled_data, columns = ['Age','Income','Expenses'])
scaled_data


# In[65]:


features = ['Age','Income']
response = ['Expenses']
x = scaled_data[features]
y = scaled_data[response]


# In[67]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)


# In[68]:


from sklearn.linear_model import LinearRegression
from sklearn import metrics


# In[70]:


model = LinearRegression()
model.fit(x_train,y_train)


# In[71]:


accuracy = model.score(x_test,y_test)
print(accuracy*100,'%')


# In[72]:


x_test


# In[73]:


y_test


# In[76]:


predicted_df = pd.DataFrame(model.predict(x_test))
predicted_df 


# In[77]:


model.intercept_


# In[80]:


model.coef_


# In[81]:


# expense = (-0.31855265*0.263736 + 0.59063665*0.381818) + model.intercept_
# expense


# In[82]:


import pickle
pickle.dump(model,open('model.pkl','wb'))


# In[84]:


model = pickle.load(open('model.pkl','rb'))
print(model.predict([[24,30000]]))


# In[ ]:





# In[ ]:




