# -*- coding: utf-8 -*-
"""
Created on Tue May 28 19:05:05 2019

@author: prateek jain
"""

# -*- coding: utf-8 -*-
"""
Created on Tue May 28 17:55:22 2019

@author: prateek jain
"""


import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import numpy as np
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression


f=pd.read_excel("sum.xlsx")
f.dropna()
df=pd.DataFrame(f,columns=['x','y','sum'],dtype=float)

#print(df)

X=df[['x','y']]
#X=X.reshape(-1,1)
Y=df['sum']
#X=X.reshape(-1,1)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2) 

##============================== Linear Regression ============================
#
clf = LinearRegression()

clf.fit(X_train, Y_train)

#testing graph

#plt.scatter(X_train['x'],Y_train,color='r')
#plt.scatter(X_train['y'],Y_train,color='y')
#plt.plot(X_train,clf.predict(X_train),color='g')

confidence = clf.score(X_test, Y_test)
print("Linear Regression : ",confidence)

print("Linear Regression : ",clf.predict([[40,34]]))

#print("intercept : ",clf.intercept_)
#print("coefficient : ",clf.coef_)