import pandas as pd
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import mean_squared_error

df = load_boston()
#print df.feature_names
boston = pd.DataFrame(df.data,columns=df.feature_names)
boston['PRICE'] = df.target


#target_names = df.target_names
#print boston

X = boston.drop('PRICE', axis=1)
y = boston['PRICE']
print X, y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

model = LinearRegression()
model.fit(X_train, y_train)

predict = model.predict(X_test)

print(np.sqrt(mean_squared_error(y_test,predict)))
