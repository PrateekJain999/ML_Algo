# -*- coding: utf-8 -*-
"""
Created on Tue May 28 18:04:06 2019

@author: prateek jain
"""

import quandl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import  model_selection, svm
from sklearn.linear_model import LinearRegression
import math

df=quandl.get("WIKI/GOOGL",authtoken='QgFsT15sDCNHg_dL-PXn')

#print(df.head())

df['HL_PCT']=(df['Adj. High']-df['Adj. Close'])/df['Adj. Close']*100.0
#print(df.head())

df['PCT_change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0
#print(df.head())

df=df[['Adj. Close','HL_PCT','PCT_change','Adj. Volume']]
#print(df.head())


forecast_col='Adj. Close'
df.fillna(value=-99999,inplace=True)
forecast_out=int(math.ceil(0.001*len(df)))

df['label']=df[forecast_col].shift(-forecast_out)
print(df.head())

df.dropna(inplace=True) 
          
X = np.array(df.drop(['label'], 1)) 
Y = np.array(df['label'])         


#====================== dividing testing and training data ==================

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2) #give 30% data for training ml clasifier and classifier is called where algo is saved  

#============================== Linear Regression ============================

clf = LinearRegression()

clf.fit(X_train, Y_train)

#testing graph

#plt.scatter(X_test,Y_test,color='r')
#plt.plot(X_train,clf.predict(X_train),color='g')

confidence = clf.score(X_train, Y_train)
print("Linear Regression : ",confidence)
print("prediction : ",clf.predict([[50.322842,8.072956,0.324968,44659000.0]]))

#print("intercept : ",clf.intercept_)
#print("coefficient : ",clf.coef_)

#================================= SVR machanism =============================#
#
#for k in ['linear','poly','rbf','sigmoid']:
#    clf = svm.SVR(kernel=k)
#    clf.fit(X_train, Y_train)
#    confidence = clf.score(X_test, Y_test)
#    print(k," : " ,confidence) 