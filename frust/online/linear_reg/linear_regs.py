import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import  matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('USA_Housing.csv')
print df.info()

X = df[['Avg. Area Income','Avg. Area House Age','Avg. Area Number of Rooms','Avg. Area Number of Bedrooms',
        'Area Population']]
y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

model = LinearRegression()
model.fit(X_train,y_train)

print(model.intercept_)
print(model.coef_)

coef_df = pd.DataFrame(model.coef_,X.columns,columns=['Coeff'])
print(coef_df)

predictions = model.predict(X_test)
#plt.scatter(y_test, predictions)

sns.distplot((y_test-predictions))
#sns.plt.show()

#evaluattion metrics
print (np.sqrt(metrics.mean_squared_error(y_test,predictions)))
