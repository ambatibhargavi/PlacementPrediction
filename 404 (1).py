#!/usr/bin/env python
# coding: utf-8

# In[1]:


# importing libraries 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn


# In[2]:


# reading a dataset
placement = pd.read_csv("Placement.csv")


# In[3]:


# reading first 10 in the dataset
placement.head(10)


# In[4]:


# defining a placement copy variable
placement_copy=placement.copy()


# In[5]:


# knowing the shape of the dataset
placement_copy.shape


# In[6]:


# to define the datatype of the dataset
placement_copy.dtypes


# In[7]:


# to define the average
placement_copy.isnull().sum()


# In[8]:


# condition for printing the axis and the required fields
placement_copy.drop(['sl_no','ssc_b','hsc_b'], axis = 1 , inplace = True)


# In[9]:


# to display the elements
placement_copy.head()


# In[10]:


# plotting the figure according to percentage
plt.figure(figsize = (15,10))

ax = plt.subplot(221)
plt.boxplot(placement_copy['ssc_p'])
ax.set_title('Secondary School Percentage')

ax = plt.subplot(222)
plt.boxplot(placement_copy['hsc_p'])
ax.set_title('Higher secondary Percentage')

ax = plt.subplot(223)
plt.boxplot(placement_copy['degree_p'])
ax.set_title('UG Percentage')

ax = plt.subplot(224)
plt.boxplot(placement_copy['etest_p'])
ax.set_title('Employability Percentage')


# In[14]:


# definig the variable
Q1 = placement_copy['hsc_p'].quantile(0.25)
Q3 = placement_copy['hsc_p'].quantile(0.75)
IQR = Q3 - Q1

filter = (placement_copy['hsc_p'] >= Q1 - 1.5 * IQR) & (placement_copy['hsc_p']<= Q3+ 1.5*IQR)
placement_filtered= placement_copy.loc[filter]


# In[13]:


# visualizing the output
plt.boxplot(placement_filtered['hsc_p'])


# In[15]:


# plotting the figure using seaborn(sns)
plt.figure(figsize = (15,7))

plt.subplot(231)
ax = sns.countplot(x= 'gender' , data = placement_filtered)

plt.subplot(232)
ax = sns.countplot(x= 'hsc_s' , data = placement_filtered)

plt.subplot(233)
ax = sns.countplot(x= 'degree_t' , data = placement_filtered)

plt.subplot(234)
ax = sns.countplot(x= 'specialisation' , data = placement_filtered)

plt.subplot(235)
ax = sns.countplot(x= 'workex' , data = placement_filtered)

plt.subplot(236)
ax = sns.countplot(x= 'status' , data = placement_filtered)


# In[16]:


# analyzing with histogram model
placement.hist()
plt.show()


# In[17]:


# pair plotting with all the models
sns.pairplot(data=placement, hue='gender')


# In[18]:


# showing corelation between the parameters
sns.heatmap(placement.corr())


# In[19]:


# graphical representation on the basis of salary
placement_placed = placement_filtered[placement_filtered.salary!= 0]
sns.distplot(placement_placed['salary'])


# In[20]:


# representation on gender 
import plotly.express as px
px.violin(placement_copy , y = 'salary' , x = 'specialisation' , color = 'gender' , box = True , points = 'all')


# In[21]:


# executing label encoding where the data is assigned with numeric values 
from sklearn.preprocessing import LabelEncoder

object_cols= ['gender','workex','specialisation','status']

label_encoder = LabelEncoder()

for col in object_cols:
    placement_filtered[col]= label_encoder.fit_transform(placement_filtered[col])
    
placement_filtered.head(10)


# In[22]:


# One Hot Encoding(creating a number of columns using dummy variable)
dummy_hsc_s = pd.get_dummies(placement_filtered['hsc_s'], prefix = 'dummy')
dummy_degree_t = pd.get_dummies(placement_filtered['degree_t'], prefix = 'dummy')

placement_coded = pd.concat([placement_filtered , dummy_hsc_s , dummy_degree_t],axis = 1)
placement_coded.drop(['hsc_s','degree_t','salary'],axis = 1 , inplace = True)
placement_coded.head()


# In[23]:


#splitting the data
X = placement_coded.drop(['status'],axis=1)
y = placement_coded.status


# In[24]:


# splitting it into train and test model
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X , y , train_size = 0.8 , random_state = 1)


# In[25]:


#using logistic regression to know the acurracy
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg = LogisticRegression()

logreg.fit(X_train , y_train)

y_pred = logreg.predict(X_test)

print(logreg.score(X_test , y_test))


# In[26]:


# using decision tree classifier to print the accuracy of the placement
from sklearn.tree import DecisionTreeClassifier 

dt = DecisionTreeClassifier(criterion = 'gini' , max_depth = 3)

dt = dt.fit(X_train , y_train)
y_pred = dt.predict(X_test)

print("Accuracy", metrics.accuracy_score(y_test , y_pred))


# In[27]:


#using random forest classsifier to print the classifier
from sklearn.ensemble import RandomForestClassifier

rt = RandomForestClassifier(n_estimators = 100)

rt.fit(X_train , y_train)
y_pred = rt.predict(X_test)

print("Accuracy", metrics.accuracy_score(y_test , y_pred))


# In[28]:


# here the placement prediction is done by showing all the model graphs, bar graphs , and also by using the regressions


# In[ ]:




