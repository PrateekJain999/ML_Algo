# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 13:45:54 2019

@author: prateek jain
"""

import numpy as np
from sklearn import preprocessing, model_selection,svm
import pandas as pd 
 
df = pd.read_csv('cancer.csv')
df.replace('?',-99999, inplace=True)
df.drop(['id'], 1, inplace=True)
df.dropna(inplace=True)
 
X = np.array(df.drop(['class'], 1))
y = np.array(df['class']) 
 
X_train, X_test, y_train, y_test = model_selection.train_test_split(X, y, test_size=0.2) 
 
clf = svm.SVC() 
clf.fit(X_train, y_train)

confidence = clf.score(X_test, y_test)
print(confidence) 
 
example_measures = np.array([[4,2,1,1,1,2,3,2,1]])
example_measures = example_measures.reshape(len(example_measures), -1)
prediction = clf.predict(example_measures)
print(prediction) 