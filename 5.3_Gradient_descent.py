# -*- coding: utf-8 -*-
"""
Created on Tue Feb 13 15:09:49 2024

@author: sahil
"""

import pandas as pd

df = pd.read_csv("C:/DS2/5_Ensemble_Techniques/movies_classification.csv")

#Dummy variables
df.head()
df.info()

# n-1 dummy variables willl be created for n categories
df = pd.get_dummies(df,columns=["3D_available","Genre"],drop_first=True)
df.head()

#Input and Output Split

predictors = df.iloc[:,df.columns!="Start_Tech_Oscar"]
type(predictors)

target = df["Start_Tech_Oscar"]
type(target)

#Train test partition of the data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(predictors,target,test_size=0.2,random_state=0)

#Refer to the links
# https://scikit-learn.org/stable/modules/classes.html

from sklearn.ensemble import GradientBoostingClassifier

boost_clf = GradientBoostingClassifier()

boost_clf.fit(x_train,y_train)

from sklearn.metrics import accuracy_score,confusion_matrix
confusion_matrix(y_test, boost_clf.predict(x_test))
accuracy_score(y_test, boost_clf.predict(x_test))


#Hyperparameters
boost_clf2 = GradientBoostingClassifier(learning_rate=0.02,n_estimators=1000,max_depth=1)
boost_clf2.fit(x_train,y_train)

from sklearn.metrics import accuracy_score,confusion_matrix

#Evaluation on testing Data
confusion_matrix(y_test, boost_clf2.predict(x_test))
accuracy_score(y_test, boost_clf2.predict(x_test))

#Evaluate on Training Data 
accuracy_score(y_train, boost_clf2.predict(x_train))
