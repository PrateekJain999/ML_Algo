import sklearn.datasets as datasets
import numpy as np
from sklearn.externals.six import StringIO
from IPython.display import Image
from sklearn.tree import export_graphviz
import pydotplus
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

df = pd.read_csv('iris.csv')

df.dropna(inplace=True)
df.drop(['Id'], 1, inplace=True) 
 
X = np.array(df.drop(['Species'], 1))
Y = np.array(df['Species'])

clf=DecisionTreeClassifier()
clf.fit(X,Y)

dot_data=StringIO()
export_graphviz(clf,out_file=dot_data,filled=True,rounded=True,special_characters=True)
graph=pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
