from sklearn import preprocessing
import pandas as pd
train = pd.DataFrame()
train_copy = train.copy()

train_copy['Cabin'] = train_copy['Cabin'].fillna('-1')

train_copy['Embarked'] = train_copy['Embarked'].fillna('-1')
cols = train_copy.columns

le = preprocessing.LabelEncoder()

for col in cols:
    train_copy[col] = le.fit_transform(train_copy[col])

train_copy.head(5)