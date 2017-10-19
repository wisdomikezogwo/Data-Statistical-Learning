import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns

ad_data = pd.read_csv('advertising.csv')

#print(ad_data.head())

def graphical_rep():

    #print(ad_data.info())
    print(ad_data.describe())
    sns.set_style('whitegrid')
    #sns.distplot(ad_data['Age'], bins=30)
    #sns.jointplot('Area Income', 'Age' , data=ad_data)
    #sns.jointplot('Age', 'Daily Time Spent on Site' , data=ad_data, kind='kde')
    sns.pairplot(ad_data,hue='Clicked on Ad')
    sns.plt.show()
    #sns.heatmap(ad_data.isnull(),yticklabels=False,cbar=False)
    #based on the heatmap , there ar no NaN

def cleanUp(data):

    data.drop(['Ad Topic Line', 'City' ,'Country', 'Timestamp'], axis= 1,inplace= True)
    print(data.head())
    return data

ad_data = cleanUp(ad_data)
print(ad_data.head())
print(ad_data.info())
X = ad_data.drop(['Clicked on Ad'], axis=1)
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=101)

model = LogisticRegression()
model.fit(X_train,y_train)

predict = model.predict(X_test)

print(classification_report(y_test,predict))
print(confusion_matrix(y_test,predict))