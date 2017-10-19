import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('/home/ikezogwo/PycharmProjects/Data-Machine-Learning/frust/online/KNN/KNN_Project_Data')
print (data.head())
sns.set_style('whitegrid')
#sns.pairplot(data,'TARGET CLASS')
#sns.plt.show()

#Standardiing the data
scaler = StandardScaler()
scaled_data = scaler.fit_transform(data.drop('TARGET CLASS', axis=1))

data_ft = pd.DataFrame(scaled_data, columns=data.columns[:-1])
print data_ft

X = data_ft
y = data['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

def best_k():
    error_rate = []
    for i in xrange(1,50):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        predict1 = knn.predict(X_test)
        error_rate.append(np.mean(predict1 != y_test))
    min_k = int(pd.DataFrame(error_rate).idxmin(axis=0)) + 1
    print min_k

    return min_k


knn = KNeighborsClassifier(n_neighbors=best_k())
knn.fit(X_train, y_train)
predict = knn.predict(X_test)

print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))
