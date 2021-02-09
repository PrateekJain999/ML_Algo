# -*- coding: utf-8 -*-
"""
Created on Fri May 31 18:05:44 2019

@author: prateek jain
"""

import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import seaborn as sb

df=pd.read_csv('poly.csv')
df.dropna(inplace=True)
sb.pairplot(df)
plt.show()

x=df.iloc[:,0:1].values
y=df.iloc[:,1:2].values

clf1=LinearRegression()
clf1.fit(x,y)

clf2=PolynomialFeatures(degree=3)
x_poly=clf2.fit_transform(x)
#print(x_poly)
clf2.fit(x_poly,y)
clf3=LinearRegression()
clf3.fit(x_poly,y)

plt.scatter(x, y, color='r',label='Data')
plt.plot(x, clf1.predict(x), color='m',label='linear')
plt.title('Linear model')

plt.plot(x, clf3.predict(clf2.fit_transform(x)), color='b',label='poly')
plt.title('polynomial model')
plt.legend()

print("Linear prediction : ",clf1.predict([[2]]))
print("Linear prediction Accuracy : ",clf3.score(x,y))
print("polynomial prediction : ",clf3.predict(clf2.fit_transform([[2]])))
