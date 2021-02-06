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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
#from matplotlib.colors import Listedcolormap

df=pd.read_csv("log.csv")
print(df)

X=df.iloc[:,2:4].values
X=preprocessing.scale(X)
Y=df.iloc[:,4].values

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2) 

##============================== Linear Regression ============================
#
clf = LogisticRegression()

clf.fit(X_train, Y_train)

#testing graph

#plt.scatter(X_train['x'],Y_train,color='r')
#plt.scatter(X_train['y'],Y_train,color='y')
#plt.plot(X_train,clf.predict(X_train),color='g')

confidence = clf.score(X_test, Y_test)
print("Logrithmic Regression : ",confidence)

pred=clf.predict(X_test)
cm=confusion_matrix(Y_test,pred)
print(cm)

#X_set,Y_set=X_test, Y_test
#X1,X2=np.meshgrid(np.arange(start=X_set[:,0].min() -1 ,stop=X_set[:,0].max()+1,step=0.01)\
#                  ,np.arange(start=X_set[:,1].min() -1 ,stop=X_set[:,1].max()+1,step=0.01))
#plt.contourf(X1,X2,clf.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),\
#             alpha=0.75,cmap=Listedcolormap(('red','green')))
#
#plt.xlim(X1.min(),X1.max())
#plt.ylim(X2.min(),X2.max())
#
#for i,j in enumerate(np.unique(Y_set)):
#    plt.scatter(X_set[Y_set == j,0],X_set[Y_set == j,1],c=Listedcolormap(('red','green'))(i),label=j)
#
#plt.title("LogisticRegression")
#plt.legend()