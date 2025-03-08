# -*- coding: utf-8 -*-
"""Spam Email Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1xr3pMyMX8KotckL7jGzDYt4aBVTowDBF
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data=pd.read_csv("mail_data.csv")

data.head()

data.shape

data.where(pd.notnull(data),"")

data.loc[data['Category'] == 'spam', 'Category',] = 0
data.loc[data['Category'] == 'ham', 'Category',] = 1

data.head()

X=data["Message"]
Y=data["Category"]

print(X)

print(Y)

X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.2,random_state=3)

print(X.shape,X_train.shape,X_test.shape)

feature_extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

X_train_feature=feature_extraction.fit_transform(X_train)
X_test_feature=feature_extraction.transform(X_test)

Y_train=Y_train.astype('int')
Y_test=Y_test.astype('int')

print(X_train)

model=LogisticRegression()

# Make sure the model is trained before predicting
model.fit(X_train_feature, Y_train)  # Train the model

# Use the correct variable name (X_train_features)
prediction_on_training_data = model.predict(X_train_feature)
accuracy_on_training_data = accuracy_score(Y_train, prediction_on_training_data)

print('Accuracy on training data:', accuracy_on_training_data)

# Use the correct variable name and remove Y_test from predict()
prediction_on_test_data = model.predict(X_test_feature)

# Compute accuracy
accuracy_on_test_data = accuracy_score(Y_test, prediction_on_test_data)

print('Accuracy on test data:', accuracy_on_test_data)

input_mail = ["I've been searching for the right words to thank you for this breather. I promise i wont take your help for granted and will fulfil my promise. You have been wonderful and a blessing at all times"]

# convert text to feature vectors
input_data_features = feature_extraction.transform(input_mail)

# making prediction

prediction = model.predict(input_data_features)
print(prediction)


if (prediction[0]==1):
  print('Ham mail')

else:
  print('Spam mail')

