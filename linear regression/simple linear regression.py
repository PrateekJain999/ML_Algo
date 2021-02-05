
# -*- coding: utf-8 -*-
"""
Created on Sun May 26 12:15:56 2019

@author: prateek jain
"""

# -*- coding: utf-8 -*-
"""
Created on Fri May 24 13:19:17 2019

@author: prateek jain
"""

import quandl
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
import math

df=quandl.get("WIKI/GOOGL",authtoken='QgFsT15sDCNHg_dL-PXn')

#print(df.head())

df=df[['Adj. Close']]


forecast_col='Adj. Close'
df.fillna(value=-99999,inplace=True)
forecast_out=int(math.ceil(0.001*len(df)))

df['label']=df[forecast_col].shift(-forecast_out)
#print(df.head())

df.dropna(inplace=True) 
          
X = df.iloc[:,0].values
Y = df.iloc[:,1].values         

X=X.reshape(-1,1)
Y=Y.reshape(-1,1)

X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y, test_size=0.2) #give 30% data for training ml clasifier and classifier is called where algo is saved  


#============================== Linear Regression ============================

clf = LinearRegression()

clf.fit(X_train, Y_train)

plt.scatter(X_test,Y_test,color='r')
plt.plot(X_train,clf.predict(X_train),color='g',markersize=5)
plt.plot(X_test,clf.predict(X_test),color='b')

confidence = clf.score(X_test, Y_test)
print("Linear Regression : ",confidence)

print("prediction : ",clf.predict([[60]]))

#print("intercept : ",clf.intercept_)
#print("coefficient : ",clf.coef_)