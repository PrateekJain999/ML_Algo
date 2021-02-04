# -*- coding: utf-8 -*-
"""
Created on Wed May 29 14:25:36 2019

@author: prateek jain
"""

import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd 
 
df = pd.read_csv('iris.csv')

df.dropna(inplace=True)
df.drop(['Id'], 1, inplace=True) 
 
X = np.array(df.drop(['Species'], 1))
y = np.array(df['Species']) 
 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2) 
 
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, y_train)
accuracy = clf.score(X_test, y_test)
print(accuracy)

prediction = clf.predict([[5.2,3.5,1.4,0.2]])
print(prediction)