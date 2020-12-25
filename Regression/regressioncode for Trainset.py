# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 13:52:52 2020

@author: olamide
"""


import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


#Loading in dataset 
dataset = pd.read_csv("Salary_Data.csv")
X = dataset.iloc[:,0].values
y = dataset.iloc[:,1].values
X = X.reshape(len(X),1)

#Splitting the Dataset in train test set

from sklearn.model_selection import train_test_split
(X_train,X_test,y_train,y_test) = train_test_split(X,y, test_size = 0.2)


#Importing the Regressor Libraries 
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVR
from sklearn.model_selection import cross_val_score
from xgboost import XGBRFRegressor
from sklearn.ensemble import AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor, ExtraTreesRegressor
import math



#iterating over each model for accuracy
models = []
models.append(('KNN', KNeighborsRegressor(n_neighbors = 5 , metric = 'minkowski',n_jobs = -1)))
models.append(('DecisionTree', DecisionTreeRegressor()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVR(gamma = 'auto')))
models.append(('RandomForest', RandomForestRegressor(n_estimators = 60)))
models.append(('Linear', LinearRegression()))
models.append(('Xgboost', XGBRFRegressor()))
models.append(("AdaBoost", AdaBoostRegressor(base_estimator = DecisionTreeRegressor(), n_estimators = 200, learning_rate = 0.001)))
models.append(("Bagging", BaggingRegressor(base_estimator = DecisionTreeRegressor(), n_estimators = 200)))
models.append(("Gradient Boosting",GradientBoostingRegressor( n_estimators = 200, learning_rate = 0.001)))
models.append(("Extra Tree",ExtraTreesRegressor( n_estimators = 100)))


#Importing stratifiedKfold and Cross_validation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
#kfold = StratifiedKFold(n_splits = 5)
from sklearn import metrics





result =[]
names =[]
# Iterating over Each classifier 
max_value = 0 
for label, model in models:
    print("Fitting "+ str(label) + " regression on the training set......Please wait..... ")
    results = cross_val_score(model, X_train, y_train)
    result.append(results)
    if results.mean() > max_value:
        max_value = results.mean()
        regressor =  model
        lb = label
        
    names.append(label)
    print("Accuracy on train set:  %0.3f Deviation: %0.3f  " %(results.mean(), results.std()))
    mode = model.fit(X_train, y_train)
    print("Accuracy on test set:", metrics.accuracy_score(y_test,mode.predict(X_test)))
    print()
    print()
    
    
print("Fitting the best model on the training set")
print("fitting "+ str(lb) + " model on the training set")
regressor.fit(X_train, y_train)
pred = regressor.predict(X_test)
print("The accuracy on the test set is %0.3f"%metrics.accuracy_score(y_test, pred))


# Visualizing the result on the training set
plt.scatter(X_train, y_train)
plt.plot(X_train,regressor.predict(X_train), color = 'red')
plt.title(str(lb) + "regresssor on the training set")
plt.show()


#Visualizing the result on the test set
plt.scatter(X_test, y_test)
plt.plot(X_test,regressor.predict(X_test))
plt.title(str(lb) + "regresssor on the test set")
plt.show()












 
# # Using ANN model

# importing keras library
from keras.models import Sequential 
from keras.layers import Dense

# # ADDING LAYERS
regressor = Sequential()
regressor.add(Dense(input_dim = 21, output_dim = 8, activation = 'relu'))
regressor.add(Dense(output_dim = 7, activation = "relu"))
regressor.add(Dense(output_dim = 6, activation = "relu"))
regressor.add(Dense(output_dim = 6, activation = "relu"))
regressor.add(Dense(output_dim = 6, activation = "sigmoid"))
regressor.add(Dense(output_dim = 6, activation = "sigmoid"))
regressor.add(Dense(output_dim = 6, activation = "sigmoid"))
regressor.add(Dense(output_dim = 1, activation = "sigmoid"))

# # #compiling the ANN model
regressor.compile(loss ='mean_squared_error', optimizer = 'adam', metrics = ['accuracy'])

# # #fitiing the model on the training  set
regressor.fit(X_train, y_train, nb_epoch = 500, batch_size = 100)

