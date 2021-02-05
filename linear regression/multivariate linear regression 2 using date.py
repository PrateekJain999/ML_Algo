# -*- coding: utf-8 -*-
"""
Created on Thu May 30 00:24:13 2019

@author: prateek jain
"""

import pandas as pd
import quandl
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression
import math
from datetime import datetime
from dateutil import parser

df=quandl.get("WIKI/GOOGL",authtoken='QgFsT15sDCNHg_dL-PXn')

#print(df.head())

df=df.reset_index(level=['Date'])
#print(df['Date'].head())  


df['HL_PCT']=(df['Adj. High']-df['Adj. Low'])/df['Adj. Close']*100.0
#print(df.head())

df['PCT_change']=(df['Adj. Close']-df['Adj. Open'])/df['Adj. Open']*100.0
#print(df.head())

df=df[['Date','Adj. Close','HL_PCT','PCT_change','Adj. Volume']]

dt = parser.parse("2004-09-30")
c=0
for i in df['Date']:
    c=c+1
    if dt == i:
        x=df.iloc[c,:].values

forecast_col='Adj. Close'
df.fillna(value=-99999,inplace=True)
forecast_out=int(math.ceil(0.001*len(df)))

df['label']=df[forecast_col].shift(-forecast_out)
#print(df.head())

df.dropna(inplace=True) 
df=df.drop(['Date'],1)
X = np.array(df.drop(['label'], 1)) 
Y = np.array(df['label'])         


#====================== dividing testing and training data ==================

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2) #give 30% data for training ml clasifier and classifier is called where algo is saved  

#================================= SVR machanism =============================#============================== Linear Regression ============================

clf = LinearRegression()

clf.fit(X_train, Y_train)

confidence = clf.score(X_test, Y_test)
print("Linear Regression : ",confidence)
print("prediction : ",clf.predict([[x[1],x[2],x[3],x[4]]]))

#print("intercept : ",clf.intercept_)
#print("coefficient : ",clf.coef_)

#testing graph

#plt.scatter(X_test,Y_test,color='r')
#plt.plot(X_train,clf.predict(X_train),color='g')