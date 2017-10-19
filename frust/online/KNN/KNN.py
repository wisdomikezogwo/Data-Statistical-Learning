import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('Classified Data', index_col=0)
print df.head()

scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))

scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))

df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
X = df_feat
y = df['TARGET CLASS']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

def best_k():
    error_rate = []
    for i in xrange(1,50):
        knn = KNeighborsClassifier(n_neighbors=i)
        knn.fit(X_train, y_train)
        predict1 = knn.predict(X_test)
        error_rate.append(np.mean(predict1 != y_test))
    min_k = pd.DataFrame(error_rate).idxmin(axis= 0)
    min_k = int(min_k)+ 1
    print min_k

    return error_rate


def plot_best_error():
    error_rate = best_k()
    plt.figure(figsize=(10,6))
    plt.plot(range(1,50), error_rate, color= 'blue')
    plt.xlabel('K')
    plt.ylabel('Error Rate')
    plt.show()

plot_best_error()


knn = KNeighborsClassifier(n_neighbors=34)
knn.fit(X_train, y_train)
predict = knn.predict(X_test)

print(confusion_matrix(y_test, predict))
print(classification_report(y_test, predict))


