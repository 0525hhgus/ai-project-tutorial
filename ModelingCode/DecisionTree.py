from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import tree
from IPython.display import  Image
import pandas as pd
import numpy as np
import pydot

data = pd.read_csv('pjdata_pre_cg2_nh.csv', names=['Tem', 'Wind', 'Hum', 'Pres',
                                                       'PM10', 'PM2.5', 'O3', 'NO2',
                                                'CO', 'SO2', 'Pat'], header=None)

X = np.array(pd.DataFrame(data, columns=['Tem', 'Wind', 'Hum', 'Pres', 'PM10', 'PM2.5', 'O3', 'NO2', 'CO', 'SO2']))
y = np.array(pd.DataFrame(data, columns=['Pat']))

X_train, X_test, y_train, y_test = train_test_split(X,y)

dt_clf = DecisionTreeClassifier(random_state=0)

dt_clf = dt_clf.fit(X_train, y_train)

dt_prediction = dt_clf.predict(X_test)

export_graphviz(dt_clf, out_file="dt1.dot",  feature_names=['Tem', 'Wind', 'Hum',
                                                                                               'Pres', 'PM10', 'PM2.5',
                                                                                               'O3', 'NO2', 'CO',
                                                                                               'SO2'],
                impurity=False, filled=True)

(graph,) = pydot.graph_from_dot_file('dt1.dot',encoding='utf8')

graph.write_png('dt1.png')

