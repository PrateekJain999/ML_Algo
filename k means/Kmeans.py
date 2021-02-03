# -*- coding: utf-8 -*-
"""
Created on Thu Jun  6 15:22:09 2019

@author: prateek jain
"""
import pandas as pd
import matplotlib.pyplot as plt 
from matplotlib import style 
import numpy as np 
from sklearn.cluster import KMeans

style.use('ggplot') 
 
#ORIGINAL: 
 
X = np.array([[1, 2],[1.5, 1.8],[5, 8],[8, 8],[1, 0.6],[9, 11]])
#df=pd.read_csv('titanic.csv')

plt.scatter(X[:, 0],X[:, 1], s=5, linewidths = 5)
plt.show()

clf = KMeans(n_clusters=3)
clf.fit(X)

centroids = clf.cluster_centers_
print(centroids)
labels = clf.labels_
print(labels)

colors = ["g.","r.","c.","y."]
for i in range(len(X)):
    plt.plot(X[i][0], X[i][1], colors[labels[i]], markersize = 10)
    
plt.scatter(centroids[:, 0],centroids[:, 1], marker = "x", s=150, linewidths = 5, zorder = 10)

plt.show()