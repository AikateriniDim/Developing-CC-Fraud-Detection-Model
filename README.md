**Using Jupyter notebook, we load a public database from Kaggle (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud), targeting to find an optimal model, which will recognise fraudulent CC transactions, overcoming the challenge of imbalanced data.**


#We install the necessary packages and import libraries.

import matplotlib.pyplot as plt 

import numpy as np

import os # accessing directory structure

import pandas as pd 

import matplotlib.pyplot as plt

import seaborn as sns 

pip install shap

import shap

import xgboost as xgb

**Importing the Data**
#importing the downloaded csv file (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) using pandas
file_path = "/Users/aikaterini/Desktop/creditcard.csv"
df = pd.read_csv(file_path)

**Exploring Dataset**
#showing the first 5 rows
df.head() 
Outcome: 31 columns, V1-V28 Transactions, Time and Amount of Transactions, Class 0 or 1 for valid or fraud transactions.

#Calculating the sum of the fraud cases
df['Class'].value_counts()
Valid cases:    284315
Fraud cases:       492

**Describing the Data**
#perform high level descriptive statistics
#find the mean, std, max
df.describe()

![Image](https://user-images.githubusercontent.com/131453473/244671467-436c61bb-5175-455e-8013-e40d72f072de.png)

Outcome: Highly imbalanced data, only 0.17% are fraudulent transactions.

The type of modelling approach in this dataset is the Supervised learning approach, more specifically, the Classification.
In this dataset, the "Class" column contains the target variable, and that indicates whether the transaction was fraudulent (Yes or no).
It predicts a categorical variable (if it's fraud yes/no), which means that's a binary classification problem. 

Before starting modeling the data, we split our data into a dataset that can be used for training of a model and be a second dataset that can be used to evaluate the effectiveness of our model after training.

#drop target var and columns necessary to train model
y = df['Class']
X = df.drop(['Class', 'Amount', 'Time'], axis = 1)

**Validation of Model**

from sklearn.model_selection import train_test_split

#stratification, to avoid all of the fraud ending up in the training set or within the test
#stratify by Y, splits or partitions the positive instances
X_train, X_test, y_train, y_test = train_test_split (X,y, test_size= 0.1, random_state = 42, stratify = y)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)
print("y_train:", y_train.shape)
print("y_test:", y_test.shape)

#number of fraudulent cases in the sample datasets
print ("Fraud in y_train:", len(np.where(y_train ==1)[0]))
print ("Fraud in y_test:", len(np.where(y_test ==1)[0]))

Fraud in y_train: 443
Fraud in y_test: 49

![Image](https://user-images.githubusercontent.com/131453473/244674321-7b1db359-0eac-4c23-90b7-a426d33b81a4.png)

**Modeling**
#Training a Baseline Logistic Regression Model
from sklearn.linear_model import LogisticRegression

#initialise model
model = LogisticRegression()

#train the model using the using the training data, partitioned in the train test
#x_train contains the features that the model is going to learn on
#y_train contains the fraudulent records which the model will learn from and correct itself from

model.fit(X_train, y_train)

#once model is trained, we can do predictions on the test set
y_pred = model.predict(X_test)
y_pred

#Confusion Matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns 

True negatives = non-fraud cases which are correctly classified as non-fraud
False positives = non-fraud cases but incorrectly classified as non-fraud
False negatives = fraud cases but incorrectly classified as non-fraud
True positives = fraud and correctly classified as fraud

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

![Image](https://user-images.githubusercontent.com/131453473/244698781-8de9ffe6-e844-4cf3-a69c-af768d18909a.png)

Model detected 34 fraudulent cases, 15 cases were not correctly predicted by the model as they were actually fraudulent, 6 cases are false positives and 28426 are the cases that the model predicts as valid and they actually are.

**Implementing Hyperparameter Selection to improve the Logistic Regression Model**
model = LogisticRegression(class_weight = 'balanced')
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred

![Image](https://user-images.githubusercontent.com/131453473/244703281-8a9f3b6d-ae90-48e1-9359-287e554105e9.png)

This resulted in detecting more fraudsters but also in misclassifying a lot more good customers, so we need to balance this.
Then, we rebalance the class weights by increasing the emphasis on the minority class by 50, indicating to the model to not pay as much attention to this class as when was balanced.

model = LogisticRegression(class_weight = {0:1,1:50})
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

This results in way less false positives by missing only one fraudulent case.

![Image](https://user-images.githubusercontent.com/131453473/244705212-4adaa7f8-611d-4e1d-ae3d-0862ae6eba97.png)

Furthermore, in order to interpret the Logistic Regression model, we look at the model coefficients, which represent the importance of each of the features in our model's ability to detect fraud or not.
model.coef_

We can also find the model intercept: 
model.intercept_

We can determine the Class probability too:

#true probabilities would require model calibration isotonic regression etc
model.predict_proba(X_test)

Importing SHAP we find the average expected marginal contribution of one feature to the model after all possible feature combinations have been considered.

import shap
shap.initjs()

explainer = shap.LinearExplainer(model, X_train)
shap_values = explainer.shap_values(X)
shap.summary_plot(shap_values,X)

The transactions (V) appear in descending order with the most significant feature from a value contribution perspective at the top.
The higher the V4 value is, the higher the feature contributes positively to the detection of fraud. Whereas the V14, the lower it is, the more contributes to the model in detecting fraud.

Moreover, gradient boosting is a supervised learning algorithm, which attempts to accurately predict a target variable by combining the estimates of a set of simpler, weaker models.

#Training an XGBoost Model
import xgboost as xgb
model = xgb.XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_matrix(y_test, y_pred)

Now, we get one false positive case less, 40 fraudulent cases and 9 fraudulent cases that we couldn't predict. 

#visualising the outcome using the confusion matrix
import matplotlib.pyplot as plt
import seaborn as sns

![Image](https://user-images.githubusercontent.com/131453473/244786803-5bcc451b-bdc1-4a61-961e-7ffd3856cb92.png)

Implementing hyperparameter optimisation to get an optimal XGBoost Model.

#Train the XGBoost classifier with the scale weight hyper parameter set to 100.
model = xgb.XGBClassifier(scale_pos_weight=100)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred

from sklearn.metrics import confusion_matrix
confusion_matrix (y_test, y_pred)

Now, we get 41 fraudulent cases and slightly more misclassifications from a false positive perspective.

![Image](https://user-images.githubusercontent.com/131453473/244788707-b6429671-953c-4885-b531-71bde56e1d61.png)

In order to improve the XGBoost model, we do another iteration of hyper parameter tuning.
We keep the scaling of the positive weight hyper parameter to 100 and setting the maximum depth hyper parameter to five to reduce overfitting.

import xgboost as xgb
model_xgb = xgb.XGBClassifier (max_depth=5, scale_pos_weight=100)
model_xgb.fit(X_train, y_train)
y_pred = model_xgb.predict(X_test)

This time, we detected two more fraudulent cases and we have less false positives.
There's only three non fraudulent transactions that we have flagged as fraudulent, which is a better result.

#visualising the outcome using the confusion matrix

![Image](https://user-images.githubusercontent.com/131453473/244792607-a5a14229-9437-4698-9d22-494552a51813.png)

In order to explain the optimal XGBoost model, we can check how how many classes the model has been trained to predict and the feature importances, which will reveal the most significant positive class to the detection.

model.classes_
model.feature_importances_

**Performance metrics for optimal XGBoost model**
Using scikit-learn we can examine various performance metrics for our optimal XGBoost model.

• Precision is the proportion of correctly predicted fraudulent cases among all predicted as fraud cases. (True Positive / True Positive + False Positive)
• Recall is the proportion of the fraudulent instances that are successfully predicted. (True Positive / True Positive + False Negative)
• F1-score is the harmonic mean of precision and recall.
• AUPRC = Area under the Precision-Recall; it includes True Negatives, which influences the scores significantly in highly imbalanced data

average_precision_score(y_test, y_pred)

#Classification report summarising the classification metrics
print(classification_report(y_test, y_pred))

![Image](https://user-images.githubusercontent.com/131453473/244804105-3cd5dfc8-e7b0-400c-b50d-43d56757e95e.png)
