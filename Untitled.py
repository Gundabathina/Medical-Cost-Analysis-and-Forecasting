#!/usr/bin/env python
# coding: utf-8

# In[17]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[24]:


pip install pandas openpyxl


# In[29]:


df = pd.read_excel(r"C:\Users\premt\OneDrive\Desktop\Medical Cost Analysis Data Set.xlsx")
df.head()


# In[32]:


df.dtypes


# In[33]:


df.isnull().sum()


# In[36]:


from sklearn.preprocessing import LabelEncoder
df_aug = pd.read_excel(r"C:\Users\premt\OneDrive\Desktop\Medical Cost Analysis Data Set.xlsx")
#sex
le = LabelEncoder()
le.fit(df_aug.sex.drop_duplicates()) 
df_aug.sex = le.transform(df_aug.sex)
# smoker or not
le.fit(df_aug.smoker.drop_duplicates()) 
df_aug.smoker = le.transform(df_aug.smoker)
#region
le.fit(df_aug.region.drop_duplicates()) 
df_aug.region = le.transform(df_aug.region)


# In[37]:


df_aug


# In[38]:


df_aug.region.value_counts()


# In[39]:


df.region.value_counts()


# In[40]:


f, ax = pl.subplots(figsize=(10, 8))
corr = df_aug.corr()
sns.heatmap(corr, cmap='coolwarm')


# In[41]:


corr['charges'].sort_values()


# In[45]:


f= pl.figure(figsize=(12,5))

ax=f.add_subplot(121)
sns.distplot(df_aug[(df.smoker == 'yes')]["charges"],color='c')
ax.set_title('Distribution of charges for smokers')

ax=f.add_subplot(122)
sns.distplot(df_aug[(df_aug.smoker == 0)]['charges'],color='b')
ax.set_title('Distribution of charges for non-smokers')


# In[46]:


sns.catplot(x="smoker", kind="count",hue = 'sex',  data=df)


# In[47]:


sns.catplot(x="sex", y="charges", hue="smoker", kind="violin", data=df)


# In[48]:


pl.figure(figsize=(12,5))
pl.title("Box plot for charges of women")
sns.boxplot(y="smoker", x="charges", data =  df_aug[(df_aug.sex == 0)] , orient="h", palette = 'magma')


# In[49]:


pl.figure(figsize=(12,5))
pl.title("Box plot for charges of men")
sns.boxplot(y="smoker", x="charges", data =  df_aug[(df_aug.sex == 0)] , orient="h", palette = 'magma')


# In[50]:


pl.figure(figsize=(12,5))
pl.title("Distribution of age")
ax = sns.distplot(df_aug["age"], color = 'b')


# In[51]:


sns.lmplot(x="age", y="charges", hue="smoker", data=df_aug, palette = 'rainbow')
ax.set_title('Smokers and non-smokers')


# In[52]:


pl.figure(figsize=(12,5))
pl.title("Distribution of bmi")
ax = sns.distplot(df.bmi, color = 'm')


# In[53]:


pl.figure(figsize=(12,5))
pl.title("Distribution of charges for patients with BMI greater than 30")
ax = sns.distplot(df[(df.bmi >= 30)]['charges'], color = 'c')


# In[54]:


pl.figure(figsize=(12,5))
pl.title("Distribution of charges for patients with BMI less than 30")
ax = sns.distplot(df[(df.bmi < 30)]['charges'], color = 'b')


# In[55]:


pl.figure(figsize=(10,6))
ax = sns.scatterplot(x='bmi',y='charges',data=df_aug,palette='magma',hue='smoker')
ax.set_title('Scatter plot of charges and bmi')

sns.lmplot(x="bmi", y="charges", hue="smoker", data=df_aug, palette = 'magma')


# In[56]:


sns.catplot(x="children", kind="count", palette="rainbow", data=df_aug)


# In[57]:


sns.catplot(x="smoker", kind="count", palette="rainbow",hue = "sex",
            data=df[(df.children > 0)])
ax.set_title('Smokers and non-smokers who have childrens')


# In[58]:


from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score,mean_squared_error,accuracy_score
from sklearn.ensemble import RandomForestRegressor


# In[59]:


df_aug


# In[60]:


x = df_aug.drop(['charges'], axis = 1)
y = df_aug.charges

x_train,x_test,y_train,y_test = train_test_split(x,y, random_state = 0)
lr = LinearRegression()
lr.fit(x_train,y_train)

y_train_pred = lr.predict(x_train)
y_test_pred = lr.predict(x_test)

print(lr.score(x_test,y_test)*100,"%")


# In[61]:


X = df_aug.drop(['charges','region'], axis = 1)
Y = df_aug.charges



quad = PolynomialFeatures (degree = 2)
x_quad = quad.fit_transform(X)

X_train,X_test,Y_train,Y_test = train_test_split(x_quad,Y, random_state = 0)

plr = LinearRegression().fit(X_train,Y_train)

Y_train_pred = plr.predict(X_train)
Y_test_pred = plr.predict(X_test)

print(plr.score(X_test,Y_test)*100,"%")


# In[ ]:




