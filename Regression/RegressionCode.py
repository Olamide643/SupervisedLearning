# -*- coding: utf-8 -*-
"""
Created on Tue Apr 14 00:54:36 2020

@author: olamide
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

dataset = pd.read_csv("Salary_Data.csv")

X = dataset.iloc[:,0].values
Y = dataset.iloc[:,1].values

from sklearn.model_selection import train_test_split
(X_train,X_test, Y_train,Y_test) = train_test_split(X,Y,test_size =1/3, random_state = None )

#reshaping the dataset 
X_train = X_train.reshape(20,1)
X_test = X_test.reshape(10,1)
Y_train = Y_train.reshape(20,1)
Y_test = Y_test.reshape(10,1)

#Using the linear Regression Model
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, Y_train)
predict = lr.predict(X_test)

#Visualizing the result on the train set
plt.scatter(X_train, Y_train)
plt.plot(X_train,lr.predict(X_train))
plt.show()


#Visualizing the result onthe test set
plt.scatter(X_test, Y_test)
plt.plot(X_test,lr.predict(X_test))
plt.show()


#Polynomial Regression
from sklearn.preprocessing import PolynomialFeatures
lr2 = LinearRegression()
poly = PolynomialFeatures(degree = 2)
X_train_poly= poly.fit_transform(X_train)
X_test_poly = poly.fit_transform(X_test)
lr2.fit(X_train_poly,Y_train)

predict2 = lr2.predict(X_test_poly)

##Visualizing the result on the train set
plt.scatter(X_train, Y_train)
plt.plot(X_train,lr2.predict(X_train_poly), color = 'red')
plt.show()


# #Visualizing the result onthe test set
plt.scatter(X_test, Y_test)
plt.plot(X_test,lr2.predict(X_test_poly))
plt.show()

# Support vector Regression
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_1 =  sc.fit_transform(X_train)
from sklearn.svm import SVR
regressor = SVR(kernel = 'linear')
regressor.fit(X_train, Y_train)
predict3 = regressor.predict(X_test)