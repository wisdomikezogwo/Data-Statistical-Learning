import pandas as pd
import  numpy as np
import matplotlib.pyplot as plt
import  seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

loans = pd.read_csv('/home/ikezogwo/PycharmProjects/Data-Machine-Learning/frust/online/tree/loan_data.csv')
#print(loans.head())
#print(df.describe())
print loans['purpose'].value_counts()

plt.figure(figsize=(10,6))
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='Credit.Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='Credit.Policy=0')
plt.legend()
plt.xlabel('FICO')
#plt.show()

plt.figure(figsize=(10,6))
loans[loans['not.fully.paid']==1]['fico'].hist(alpha=0.5,color='blue',
                                              bins=30,label='not.fully.paid=1')
loans[loans['not.fully.paid']==0]['fico'].hist(alpha=0.5,color='red',
                                              bins=30,label='not.fully.paid=0')
plt.legend()
plt.xlabel('FICO')
#plt.show()

plt.figure(figsize=(11,7))
sns.countplot(x='purpose',hue='not.fully.paid',data=loans,palette='Set1')
#sns.plt.show()

def getting_dummies(loans):
    purpose = pd.get_dummies(loans['purpose'], drop_first=True)
    df = pd.concat([loans, purpose], axis=1)
    return df

train = getting_dummies(loans)
print train.head()
X = train.drop(['not.fully.paid', 'purpose'],axis=1)
y = train['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

rfc = RandomForestClassifier(n_estimators=600)
dtc = DecisionTreeClassifier()

rfc.fit(X_train, y_train)
dtc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)
dtc_pred = dtc.predict(X_test)

print(classification_report(y_test, rfc_pred))
print(classification_report(y_test, dtc_pred))

print(confusion_matrix(y_test, rfc_pred))
print(confusion_matrix(y_test, dtc_pred))
