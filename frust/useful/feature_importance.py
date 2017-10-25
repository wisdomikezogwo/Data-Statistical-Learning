from sklearn.ensemble import RandomForestClassifier
import pandas as pd
train = pd.DataFrame()
train_copy = train.copy()
clf = RandomForestClassifier()

X = train_copy.drop(['PassengerId','Survived'], axis=1)
y = train_copy['Survived']

features = X.columns

clf.fit(X, y)

clf.feature_importances_