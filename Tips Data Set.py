#!/usr/bin/env python
# coding: utf-8

# In[57]:


import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV,cross_val_score
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report, accuracy_score


# In[59]:


# Load the Tips dataset using Seaborn
tips_df = sns.load_dataset("tips")
tips_df


# In[60]:


# Convert categorical variables to dummy variables
tips_df = pd.get_dummies(tips_df, columns=['sex', 'smoker', 'day', 'time'], drop_first=True)


# In[61]:


# Define feature columns and target variable
# For this example, we'll predict whether the tip is above the median
tips_df['tip_above_median'] = (tips_df['tip'] > tips_df['tip'].median()).astype(int)
feature_columns = tips_df.drop(columns=['tip', 'tip_above_median'])
X = feature_columns
y = tips_df['tip_above_median']


# In[62]:


X


# In[63]:


y


# In[64]:


#Train Test Split
X_train,X_test,y_train,y_test=train_test_split(X,y,random_state=23,test_size=0.3)


# In[65]:


# Hyperparameter tuning using GridSearchCV
params = {
    'n_estimators': [50, 100, 200],
    'max_features': ['auto', 'sqrt', 'log2'],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}


# In[66]:


gcv = GridSearchCV(estimator=RandomForestClassifier(random_state=42), param_grid=params, n_jobs=-1, verbose=2)
gcv.fit(X_train, y_train)


# In[67]:


# Best model from GridSearch
best_rf = gcv.best_estimator_
best_rf


# In[68]:


# Model evaluation
y_pred = best_rf.predict(X_test)


# In[69]:


# Calculate accuracy score
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# In[70]:


# Visualization: Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='g', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()


# In[71]:


# Visualization: ROC Curve
y_prob = best_rf.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# In[72]:


# Model Performance Metrics
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[73]:


# Distribution of the target variable
plt.figure(figsize=(8, 6))
sns.countplot(x='tip_above_median', data=tips_df)
plt.title('Distribution of Tip Above Median')
plt.show()


# In[74]:


# Pairplot to visualize the pairwise relationships
sns.pairplot(tips_df, hue='tip_above_median')
plt.show()


# In[ ]:




