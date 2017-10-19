import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics
import  matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv('Ecommerce Customers')
print(customers.head())
print customers.info(), customers.describe()

#sns.jointplot(customers['Time on App'], customers['Length of Membership'],kind='hex')
#sns.pairplot(customers)
#Length of Mem
#sns.lmplot('Yearly Amount Spent', 'Length of Membership',customers)
#

X = customers[['Avg. Session Length','Time on App','Time on Website','Length of Membership']]
y = customers['Yearly Amount Spent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
model = LinearRegression()
model.fit(X_train,y_train)

print model.coef_

#predicting
predict = model.predict(X_test)

#plt.scatter(y_test,predict)

#evaluating
print(np.sqrt(metrics.mean_squared_error(y_test,predict)))

#printing residuals

coef_df = pd.DataFrame(model.coef_,X.columns,columns=['Coeffecient'])
print(coef_df)

sns.distplot((y_test-predict), bins = 50)
sns.plt.show()

#App
