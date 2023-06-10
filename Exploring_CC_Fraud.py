#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt # plotting
import numpy as np # linear algebra
import os # accessing directory structure
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

#importing csv file
file_path = "/Users/aikaterini/Desktop/creditcard.csv"
df = pd.read_csv(file_path)

#showing the first 5 rows
df.head()


# In[2]:


#Class col shows the fraudulent cases (1)
#Calculating the sum of the fraud cases

df['Class'].value_counts()


# In[3]:


#perform high level descriptive statistics
#find the mean, std, max

df.describe()


# In[4]:


#partition dataset into two different sets
#one that can be used to train the model and
#a dataset to evaluate the effectiveness of our model after training
#training the model dataset contains our target variable only
#assigning variable y


# In[5]:


y = df['Class']
X = df.drop(['Class', 'Amount', 'Time'], axis = 1)
#drop target var and columns necessary to train model


# In[6]:


from sklearn.model_selection import train_test_split


# In[7]:


#stratification, to avoid all of the fraud ending up in either
#the training set or within the test

#stratify by Y, splits or partitions the positive instances
#according to the distribution across the entire data set

X_train, X_test, y_train, y_test = train_test_split (X,y, test_size= 0.1, random_state = 42, stratify = y)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)


# In[8]:


X_train


# In[9]:


#number of fraudulent cases in the sample datasets

print ("Fraud in y_train:", len(np.where(y_train ==1)[0]))
print ("Fraud in y_test:", len(np.where(y_test ==1)[0]))


# In[10]:


#Training a Baseline Logistic Regression Model

from sklearn.linear_model import LogisticRegression

#initialize model
model = LogisticRegression()

#train the model using the using the training data, partitioned in the train test
#x_train contains the features that the model is going to learn on
#y_train contains the fraudulent records which the model will learn from and correct itself from

model.fit(X_train, y_train)


#once model is trained, we can do predictions on the test set
y_pred = model.predict(X_test)
y_pred


# In[11]:


#Confusion Matrix

from sklearn.metrics import confusion_matrix

confusion_matrix (y_test, y_pred)


# In[12]:


import matplotlib.pyplot as plt
import seaborn as sns 

LABELS = ["Valid", "Fraud"]

conf_matrix = confusion_matrix (y_test, y_pred)

plt.figure (figsize=(5,5))

sns.heatmap (conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt ="d")
plt.title ("Confusion matrix")

plt.ylabel ('True class')
plt.xlabel ('Predicted class')

plt.show()


# In[13]:


#Hyperparameter Selection

model = LogisticRegression(class_weight = 'balanced')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_pred


from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)


# In[14]:


#visualize confusion matrix with auto balanced class

LABELS = ["Valid", "Fraud"]

conf_matrix = confusion_matrix (y_test, y_pred)

plt.figure (figsize=(5,5))

sns.heatmap (conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt ="d")
plt.title ("Confusion matrix")

plt.ylabel ('True class')
plt.xlabel ('Predicted class')

plt.show()


# In[15]:


#rebalancing the class weights
#increase the emphasis on the minority class by 50
#indicating to the model to not pay as much attention to this class as when was balanced

model = LogisticRegression(class_weight = {0:1,1:50})
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

LABELS = ["Valid", "Fraud"]

conf_matrix = confusion_matrix (y_test, y_pred)
plt.figure (figsize=(5,5))

sns.heatmap (conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt ="d")
plt.title ("Confusion matrix")

plt.ylabel ('True class')
plt.xlabel ('Predicted class')

plt.show()


# In[17]:


#exploring and interpreting the Logistic Regression Model

model = LogisticRegression(class_weight = {0:1,1:50})
model.fit(X_train, y_train)


# In[18]:


model.classes_


# In[19]:


#calculating the importance of each of the features in the models' ability
#to detect fraud effectlively -> coefficients

model.coef_


# In[20]:


model.intercept_


# In[22]:


pip install shap


# In[23]:


#This gives us the average expected marginal contribution of one feature
#after all possible feature combinations have been considered.

import shap
shap.initjs()


# In[29]:


explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values,X)

#Features have the V notation in descending order with the most significant feature
#from a value contribution perspective at the top.


# In[24]:


#Training the XGBoost Model

#Gradient boosting is a supervised learning algorith,
#which attemts to accurately predict a target variable
#by combining the estimates of a set of simpler, weaker models.


# In[25]:


import xgboost as xgb

model = xgb.XGBClassifier()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)


# In[26]:


from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, y_pred)


# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns

LABELS = ["Valid", "Fraud"]

conf_matrix = confusion_matrix (y_test, y_pred)
plt.figure (figsize=(6,6))

sns.heatmap (conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt ="d")
plt.title ("Confusion matrix")

plt.ylabel ('True class')
plt.xlabel ('Predicted class')

plt.show()


# In[28]:


#Implementing hyperparameter optimisation to get an optimal XGBoost Model.
#Train the xgboost classifier with the scale weight hyper parameter set to 100

model = xgb.XGBClassifier(scale_pos_weight=100)

model.fit(X_train, y_train)


# In[29]:


y_pred = model.predict(X_test)

y_pred


# In[30]:


from sklearn.metrics import confusion_matrix

confusion_matrix (y_test, y_pred)


# In[31]:


import matplotlib.pyplot as plt
import seaborn as sns

LABELS = ["Valid", "Fraud"]

conf_matrix = confusion_matrix (y_test, y_pred)
plt.figure (figsize=(5,5))

sns.heatmap (conf_matrix, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt ="d")
plt.title ("Confusion matrix")

plt.ylabel ('True class')
plt.xlabel ('Predicted class')

plt.show()


# In[32]:


#We keep the scaling of the positive weight hyper parameter to 100,
#introducing maximum depth hyper parameter and setting it to five.

#max_depth specifies how deep each of those trees will be built.
#It reduces overfitting. 

import xgboost as xgb
model_xgb = xgb.XGBClassifier (max_depth=5, scale_pos_weight=100)
model_xgb.fit(X_train, y_train)
y_pred = model_xgb.predict(X_test)


# In[33]:


LABELS = ["Valid", "Fraud"]

conf_matrix_xgb = confusion_matrix (y_test, y_pred)
plt.figure (figsize=(6,6))

sns.heatmap (conf_matrix_xgb, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt ="d")
plt.title ("Confusion matrix")

plt.ylabel ('True class')
plt.xlabel ('Predicted class')

plt.show()


# In[34]:


#Checking how many classes the model has been trained to predict

model.classes_


# In[35]:


model.feature_importances_


# In[43]:


#implementing performance metrics
#calculating precision score

from sklearn.metrics import (classification_report, precision_score, recall_score,
                            average_precision_score, roc_auc_score,
                             f1_score, matthews_corrcoef)


# In[49]:


#Classification report summarizes the classification metrics at the class and overall level

print(classification_report(y_test, y_pred))

