import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.svm import SVC




iris_sns = sns.load_dataset("iris")



def rename(data):
    for i in data:
        if i == 0:
            return 'setosa'
        if i == 1:
            return 'versicolour'
        else:
            return 'virginica'

#terget_df = target.apply(rename,axis= 1)
##iris = pd.concat([iris,tar],axis=1)


sns.set_style('whitegrid')
#setosa = iris[iris['target']==0]

#sns.pairplot(iris_sns,hue= 'species', palette='Dark2')
sns.plt.show()

setosa = iris_sns[iris_sns['species']=='setosa']
sns.kdeplot( setosa['sepal_width'], setosa['sepal_length'],
                 cmap="plasma", shade=True, shade_lowest=False)
#sns.plt.show()
#sns.kdeplot(iris_df['sepal width (cm)'],iris_df['sepal length (cm)'], cmap='Blues',shade=True, shade_lowest=False)
#sns.kdeplot(setosa.sepal_width, setosa.sepal_length,cmap="Reds", shade=True, shade_lowest=False)

#sns.plt.show()

X = iris_sns.drop('species', axis=1)
y = iris_sns['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

svc_model = SVC()
svc_model.fit(X_train, y_train)

predictions = svc_model.predict(X_test)
print(confusion_matrix(y_test,predictions))
print(classification_report(y_test,predictions))


#GRID SEARCH

param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]}
grid = GridSearchCV(SVC(), param_grid, refit= True, verbose=2)
grid.fit(X_train, y_train)
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))
print(classification_report(y_test,grid_predictions))