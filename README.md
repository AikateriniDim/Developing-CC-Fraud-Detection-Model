This project focuses on creating a robust credit card fraud detection model using Python and various machine learning techniques.
It begins by importing necessary libraries and loading the dataset from Kaggle, which contains transaction data that we labeled as fraudulent or valid.

The dataset analysis reveals a highly imbalanced dataset, with only 0.17% of transactions being fraudulent, setting the stage for a challenging classification problem.

The project utilizes a supervised learning approach for classification, splitting the data into training and testing sets while ensuring the stratification of fraudulent cases in both sets.
Initial efforts involve training a baseline Logistic Regression model, evaluating its performance with a confusion matrix, and attempting hyperparameter selection to improve it.

To address the imbalance issue, class weights are adjusted to give more importance to the minority class, reducing false positives.
Model interpretability is enhanced by analyzing model coefficients and using SHAP values to understand feature importance.

The project then explores the power of gradient boosting by training an XGBoost model, further enhancing fraud detection performance. Hyperparameter optimization is conducted to fine-tune the XGBoost model, achieving better results in terms of detecting fraudulent cases while minimizing false positives.

Performance metrics such as precision, recall, F1-score, and AUPRC are assessed using scikit-learn, and a classification report summarizes these metrics for the optimal XGBoost model.

Overall, this project showcases a comprehensive approach to credit card fraud detection, from data exploration to model training and evaluation, emphasizing the importance of balancing model accuracy and minimizing false positives in a highly imbalanced dataset.

                                                        ********
We install the necessary packages and import libraries.

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
Importing the downloaded csv file (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) using pandas

**Exploring Dataset**
showing the first 5 rows
Outcome: 31 columns, V1-V28 Transactions, Time and Amount of Transactions, Class 0 or 1 for valid or fraud transactions.

Calculating the sum of the fraud cases

Valid cases:    284315
Fraud cases:       492

**Describing the Data**
Perform high level descriptive statistics

![Image](https://user-images.githubusercontent.com/131453473/244671467-436c61bb-5175-455e-8013-e40d72f072de.png)

Outcome: Highly imbalanced data, only 0.17% are fraudulent transactions.

The type of modelling approach in this dataset is the Supervised learning approach, more specifically, the Classification.
In this dataset, the "Class" column contains the target variable, and that indicates whether the transaction was fraudulent (Yes or no).
It predicts a categorical variable (if it's fraud yes/no), which means that's a binary classification problem. 

Before starting modeling the data, we split our data into a dataset that can be used for training of a model and be a second dataset that can be used to evaluate the effectiveness of our model after training.

**Validation of Model**

Stratification, to avoid all of the fraud ending up in the training set or within the test
Stratify by Y.

Number of fraudulent cases in the sample datasets
Fraud in y_train: 443
Fraud in y_test: 49

![Image](https://user-images.githubusercontent.com/131453473/244674321-7b1db359-0eac-4c23-90b7-a426d33b81a4.png)

**Modeling**
Training a Baseline Logistic Regression Model

Train the model using the using the training data, partitioned in the train test
x_train contains the features that the model is going to learn on
y_train contains the fraudulent records which the model will learn from and correct itself from

Once model is trained, we can do predictions on the test set

Confusion Matrix

True negatives = non-fraud cases which are correctly classified as non-fraud
False positives = non-fraud cases but incorrectly classified as non-fraud
False negatives = fraud cases but incorrectly classified as non-fraud
True positives = fraud and correctly classified as fraud

![Image](https://user-images.githubusercontent.com/131453473/244698781-8de9ffe6-e844-4cf3-a69c-af768d18909a.png)

Model detected 34 fraudulent cases, 15 cases were not correctly predicted by the model as they were actually fraudulent, 6 cases are false positives and 28426 are the cases that the model predicts as valid and they actually are.

**Implementing Hyperparameter Selection to improve the Logistic Regression Model**

![Image](https://user-images.githubusercontent.com/131453473/244703281-8a9f3b6d-ae90-48e1-9359-287e554105e9.png)

This resulted in detecting more fraudsters but also in misclassifying a lot more good customers, so we need to balance this.
Then, we rebalance the class weights by increasing the emphasis on the minority class by 50, indicating to the model to not pay as much attention to this class as when was balanced.

This results in way less false positives by missing only one fraudulent case.

![Image](https://user-images.githubusercontent.com/131453473/244705212-4adaa7f8-611d-4e1d-ae3d-0862ae6eba97.png)

Furthermore, in order to interpret the Logistic Regression model, we look at the model coefficients, which represent the importance of each of the features in our model's ability to detect fraud or not.

We can also find the model intercept.

Importing SHAP to find the average expected marginal contribution of one feature to the model after all possible feature combinations have been considered.

The transactions (V) appear in descending order with the most significant feature from a value contribution perspective at the top.
The higher the V4 value is, the higher the feature contributes positively to the detection of fraud. Whereas the V14, the lower it is, the more contributes to the model in detecting fraud.

Moreover, gradient boosting is a supervised learning algorithm, which attempts to accurately predict a target variable by combining the estimates of a set of simpler, weaker models.

Training an XGBoost Model
Now, we get one false positive case less, 40 fraudulent cases and 9 fraudulent cases that we couldn't predict. 

Visualising the outcome using the confusion matrix
![Image](https://user-images.githubusercontent.com/131453473/244786803-5bcc451b-bdc1-4a61-961e-7ffd3856cb92.png)

**Implementing hyperparameter optimisation to get an optimal XGBoost Model**

Train the XGBoost classifier with the scale weight hyper parameter set to 100.
Now, we get 41 fraudulent cases and slightly more misclassifications from a false positive perspective.

![Image](https://user-images.githubusercontent.com/131453473/244788707-b6429671-953c-4885-b531-71bde56e1d61.png)

In order to improve the XGBoost model, we do another iteration of hyper parameter tuning.
We keep the scaling of the positive weight hyper parameter to 100 and setting the maximum depth hyper parameter to five to reduce overfitting.

This time, we detected two more fraudulent cases and we have less false positives.
There's only three non fraudulent transactions that we have flagged as fraudulent, which is a better result.

Visualising the outcome using the confusion matrix
![Image](https://user-images.githubusercontent.com/131453473/244792607-a5a14229-9437-4698-9d22-494552a51813.png)

In order to explain the optimal XGBoost model, we can check how how many classes the model has been trained to predict and the feature importances, which will reveal the most significant positive class to the detection.

**Performance metrics for optimal XGBoost model**

Using scikit-learn we can examine various performance metrics for our optimal XGBoost model.

• Precision is the proportion of correctly predicted fraudulent cases among all predicted as fraud cases. (True Positive / True Positive + False Positive)
• Recall is the proportion of the fraudulent instances that are successfully predicted. (True Positive / True Positive + False Negative)
• F1-score is the harmonic mean of precision and recall.
• AUPRC = Area under the Precision-Recall; it includes True Negatives, which influences the scores significantly in highly imbalanced data

Classification report summarising the classification metrics:
![Image](https://user-images.githubusercontent.com/131453473/244804105-3cd5dfc8-e7b0-400c-b50d-43d56757e95e.png)
