# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 11:20:50 2020

@author: olamide
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


#Importing the dataset
dataset = pd.read_csv("Social_Network_Ads.csv")
X = dataset.iloc[:,1:4].values
y = dataset.iloc[:, 4].values

# # Encoding the categorical data
from sklearn.preprocessing import LabelEncoder
lb = LabelEncoder()
X[:,0] = lb.fit_transform(X[:,0])


# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X= sc.fit_transform(X)

# spliting the dataset in test_train 
from sklearn.model_selection import train_test_split
(X_train, X_test, y_train, y_test)= train_test_split(X,y, test_size = 0.2, random_state = 2019)


# Importing all the models 
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from xgboost import XGBClassifier



#Empty classifier List 
clf = []

# Defining all Classifiers with Names 
clf.append(("Logistic Regression", LogisticRegression(C = 10)))
clf.append(("Logistic Regression with C= 100", LogisticRegression(C = 100)))
clf.append(("KNN",KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p =2)))
clf.append(("DecisionTree", DecisionTreeClassifier(criterion = "entropy")))
clf.append(("Random Forest", RandomForestClassifier(n_estimators = 100, criterion = "entropy")))
clf.append(("SVC(linear)", SVC(kernel = "linear")))
clf.append(("SVC(rbf)", SVC(kernel = "rbf")))
clf.append(("SVC(sigmoid)", SVC(kernel = "sigmoid")))
clf.append(("SVC(poly)", SVC(kernel = "poly", degree = 2 )))
clf.append(("AdaBoost", AdaBoostClassifier(base_estimator = DecisionTreeClassifier(criterion = "entropy"), n_estimators = 200, learning_rate = 0.01)))
clf.append(("Bagging", BaggingClassifier(base_estimator = DecisionTreeClassifier(criterion = "entropy"), n_estimators = 200)))
clf.append(("Gradient Boosting",GradientBoostingClassifier( n_estimators = 200, learning_rate = 0.01)))
clf.append(("Extra Tree",ExtraTreesClassifier( n_estimators = 100)))
clf.append(("XGBoost", XGBClassifier()))

#Importing stratifiedKfold and Cross_validation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
kfold = StratifiedKFold(n_splits = 5)
from sklearn import metrics

#List of all accuraries over each stratas and names  
result =[]
names =[]
# Iterating over Each classifier 
max_value = 0 
for label, model in clf:
    print("Fitting "+ str(label) + " model on the training set......Please wait..... ")
    results = cross_val_score(model, X_train, y_train, cv = kfold, n_jobs = -1)
    result.append(results)
    if results.mean() > max_value:
        max_value = results.mean()
        classifier =  model
        lb = label
        
    names.append(label)
    print("Accuracy on train set:  %0.3f Deviation: %0.3f  " %(results.mean(), results.std()))
    mode = model.fit(X_train, y_train)
    print("Accuracy on test set:", metrics.accuracy_score(y_test,mode.predict(X_test)))
    print()
    print()
    
    
print("Fitting the best model on the training set")
print("fitting "+ str(lb) + " model on the training set")
classifier.fit(X_train, y_train)
pred = classifier.predict(X_test)
cm = metrics.confusion_matrix(y_test, pred)
print("The accuracy on the test set is %0.3f"%metrics.accuracy_score(y_test, pred))
