# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 14:40:56 2024

@author: sahil
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from scipy.stats import skew

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split,cross_val_predict,KFold
from sklearn.metrics import accuracy_score,classification_report

data = pd.read_csv("C:/DS2/5_Ensemble_Techniques/movies_classification.csv")

# Data information
data.head()
data.info()
data.isna().sum()
#EDA
target = data["Start_Tech_Oscar"]
sns.countplot(x = target, palette='winter')
plt.xlabel("Oscar Rate")
# data is evenly distributed
plt.figure(figsize=(16,8))
#sns.heatmap(data.corr(), annot=True,cmap= 'YlGnBu', fmt= '.2f')


# Assuming 'data' is your DataFrame
numeric_columns = data.select_dtypes(include=np.number).columns
numeric_data = data[numeric_columns]

# Create a heatmap using only numeric columns
sns.heatmap(numeric_data.corr(), annot=True, cmap='YlGnBu', fmt='.2f')
plt.show()

# Observation:
    
#1 : Lead_actor_Rating, Lead_Actress_rating Director_rating and producer rating

sns.countplot(x = 'Genre', data=data,hue='Start_Tech_Oscar',palette='pastel')
plt.title('O chance based on Ticket Class', fontsize=10);
#Observation:
## there are more chances of getting Oscar in Drama, comedy and Action
sns.countplot(x = '3D_available', data=data,hue='Start_Tech_Oscar',palette='pastel')
plt.title('O chance based on Ticket Class', fontsize=10);
#Observation:
# it is clear from the plot that if 30 is available there is chance
sns.set_context('notebook',font_scale=1.2)
fig, ax = plt.subplots(2,figsize = (20,13))

plt.suptitle('Distribution of Twitter_hastags and Collection based on target variable',fontsize=20)
axl = sns.histplot(x = 'Twitter_hastags',data=data,hue="Start_Tech_Oscar",kde=True,ax=ax[0],palette='winter')    

axl.set(xlabel = 'Twitter_hastags',title = 'Distribution of Twitter_hastags and Collection based on target variable')

ax2 = sns.histplot(x = 'Collection',data=data,hue="Start_Tech_Oscar",kde=True,ax=ax[1],palette='viridis')

ax2.set(xlabel = 'Collection',title = 'Distribution of Twitter_hastags and Collection based on target variable')

plt.show()

data.hist(bins = 30,figsize=(20,15),color= '#005b96');

#There are outliers in Twitter_hastag,marketing_expense,time_taken

sns.boxenplot(x=data["Twitter_hastags"])
sns.boxenplot(x=data["Marketing expense"])
sns.boxenplot(x=data["Time_taken"])
sns.boxenplot(x=data["Avg_age_actors"])


# write code for winsorizer
# checking skewness
numeric_columns = data.select_dtypes(include=np.number).columns
skew_df = pd.DataFrame({'Feature': numeric_columns})
#skew_df = pd.DataFrame(data.select_dtype(np.number).columns,columns=["Feature"])
skew_df['Skew'] = skew_df['Feature'].apply(lambda feature: skew(data[feature]))
skew_df['Absolute Skew'] = skew_df['Skew'].apply(abs)
skew_df['Skewed'] = skew_df['Absolute Skew'].apply(lambda x: True if x>=0.5 else False )
skew_df
# Total charges column is clearly skewed as we also saw in the histogram ,
for column in skew_df.query("Skewed == True")['Feature'].values:data[column] = np.log1p(data[column])

data.head()
#Encoding
data1 = data.copy()
data1 = pd.get_dummies(data1)

data1.head()
# Scaling
data2 = data1.copy()
sc = StandardScaler()

numeric_columns = data2.select_dtypes(np.number).columns
sc = StandardScaler()
data2[numeric_columns] = sc.fit_transform(data2[numeric_columns])

#data2[data1.select_dtypes(np.number).columns] = sc.fit_transform(data2[data])
#data2.drop(['Start_Tech_Oscar'],axis = 1,inplace = True)
#data2 = data2.drop(['Start_Tech_Oscar'], axis=1)

data2.head()

#Splitting 
data_f = data2.copy()
target = data["Start_Tech_Oscar"]
target = target.astype(int)
target
X_train, X_test, y_train,y_test = train_test_split(data_f,target,test_size = 0.2,stratify=target,random_state=42)

# Modelling
from sklearn.ensemble import AdaBoostClassifier

ada_clf = AdaBoostClassifier(learning_rate=0.02,n_estimators=5000)

ada_clf.fit(X_train,y_train)

from sklearn.metrics import accuracy_score, confusion_matrix

#Evaluation on Testing Data
confusion_matrix(y_test,ada_clf.predict(X_test))
accuracy_score(y_test, ada_clf.predict(X_test))


