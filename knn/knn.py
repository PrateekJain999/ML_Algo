# -*- coding: utf-8 -*-
"""
Created on Wed May 29 13:42:07 2019

@author: prateek jain
"""

import numpy as np
from sklearn import preprocessing, model_selection, neighbors
import pandas as pd
import matplotlib.pyplot as plt 

df = pd.read_csv('cancer.csv')
df.replace('?',-99999, inplace=True)

df.dropna(inplace=True)
df.drop(['id'], 1, inplace=True) 
 
X = np.array(df.drop(['class'], 1))
#X=preprocessing.scale(X)
Y = np.array(df['class']) 

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2) 
 
clf = neighbors.KNeighborsClassifier()
clf.fit(X_train, Y_train)
accuracy = clf.score(X_test, Y_test)
print(accuracy)

example_measures = np.array([5,2,1,1,1,2,3,2,1],dtype=float) 
#example_measures=preprocessing.scale(example_measures)
example_measures = example_measures.reshape(1, -1)
prediction = clf.predict(example_measures)
print(prediction)


#plt.scatter(X_test,Y_test,color='r')
#plt.plot(X_train,clf.predict(X_train),color='g')

