import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
#print test.head()
#print train.head()
def input_age (data):
    Age = data[0]
    Pclass = data[1]

    if pd.isnull(Age):
        if Pclass == 1:
            return 37
        if Pclass == 2:
            return 29
        else :
            return 24
    else:
        return Age

train['Age'] = train[['Age','Pclass']].apply(input_age, axis=1)
train.drop('Cabin',axis=1,inplace=True)
test['Age'] = test[['Age','Pclass']].apply(input_age, axis=1)
test.drop('Cabin',axis=1,inplace=True)


def input_fare_test(data):
    Fare = data[0]
    Pclass = data[1]

    if pd.isnull(Fare):
        if Pclass == 1:
            return 59
        if Pclass == 2:
            return 19
        else:
            return 10
    else:
        return Fare


test['Fare'] = test[['Fare', 'Pclass']].apply(input_fare_test, axis=1)


def getting_dummies(data):
    Sex = pd.get_dummies(data['Sex'], drop_first=True)
    Embark = pd.get_dummies(data['Embarked'],drop_first=True)
    data = pd.concat([data,Sex,Embark], axis=1)
    return data

train = getting_dummies(train)
test = getting_dummies(test)

train.drop(['Sex','Embarked','Ticket','Name','PassengerId'],axis=1,inplace=True)
test.drop(['Sex','Embarked','Ticket','Name','PassengerId'],axis=1,inplace=True)

X = train.drop('Survived', axis=1)
y = train['Survived']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

model = LogisticRegression()
model.fit(X_train,y_train)

predict = model.predict(X_test)
#predict1 = model.predict(test)

print(classification_report(y_test,predict))
print(confusion_matrix(y_test,predict))
print '                             '
#print(classification_report(y_test,predict1))
#print(confusion_matrix(y_test,predict1))
#print X_test.head()
#print test.head()


sns.set_style('whitegrid')
#sns.countplot('Survived',hue='Pclass',data=train, palette='RdBu_r')
#sns.distplot(train['Age'].dropna(), kde=False,bins = 30)
#sns.heatmap(test.isnull(),yticklabels=False,cbar=False)
#sns.boxplot(x='Pclass', y='Fare', data=train)
sns.plt.show()


