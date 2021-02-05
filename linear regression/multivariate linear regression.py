# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:55:22 2019

@author: prateek jain
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn import  model_selection
from sklearn.linear_model import LinearRegression


f=pd.read_csv("Advertising.csv")
f.dropna()
df=pd.DataFrame(f,columns=['TV','radio','newspaper','sales'],dtype=float)

#print(df)

X=df[['TV','radio','newspaper']]
#X=X.reshape(-1,1)
Y=df['sales']
#X=X.reshape(-1,1)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2) 

##============================== Linear Regression ============================
#
clf = LinearRegression()

clf.fit(X_train, Y_train)

#testing graph

#plt.scatter(X_test,Y_test,color='r')
#plt.plot(X_train,clf.predict(X_train),color='g')

confidence = clf.score(X_train, Y_train)
print("Linear Regression : ",confidence)

print("prediction : ",clf.predict([[44.4,39.3,45.1]]))

#print("intercept : ",clf.intercept_)
#print("coefficient : ",clf.coef_)