import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report

cancer = load_breast_cancer()
print cancer.keys()
print cancer.DESCR

df_feat = pd.DataFrame(data=cancer['data'], columns=cancer['feature_names'])
print df_feat.head()

X = df_feat
y=cancer.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

model = SVC()
model.fit(X_train, y_train)
l = model.get_params()
print l
predict = model.predict(X_test)



print (confusion_matrix(y_test, predict))
print('\n')
print (classification_report(y_test, predict))

#Using Grid serach
param_grid = {'C':[0.1,1,10,100,1000], 'gamma':[1,0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(SVC(),param_grid,verbose=3)
grid.fit(X_train, y_train)

print grid.best_params_
print grid.best_score_ , grid.best_estimator_

grid_predict = grid.predict(X_test)

print (confusion_matrix(y_test, grid_predict))
print('\n')
print (classification_report(y_test, grid_predict))
