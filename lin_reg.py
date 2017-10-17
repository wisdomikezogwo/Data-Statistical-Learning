import pandas as pd
import numpy as np
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import math, datetime, time
import quandl as q
import matplotlib.pyplot as plt
from matplotlib import style
import pickle
style.use('ggplot')


df = q.get('WIKI/GOOGL')

df = df[['Adj. Open','Adj. High','Adj. Low','Adj. Close','Adj. Volume',]]
#percent voaitlity
df['HL_PER'] = ((df['Adj. High'] - df['Adj. Close'])/ df['Adj. Close'] )*100

#percent voaitlity
df['PER_Chg'] = ((df['Adj. Close'] - df['Adj. Open'])/ df['Adj. Open'] )*100
df = df[ ['Adj. Close', 'HL_PER', 'PER_Chg', 'Adj. Volume']]

forecast_col = 'Adj. Close'
df.fillna(-99999, inplace=True)

forecast_out  = int(math.ceil(0.1*len(df)))
print (forecast_out)
#creating label

df['label']= df[forecast_col].shift(-forecast_out)

#x -features
#y- labels

X = np.array(df.drop(['label'], 1))
#scale X you have to scale alongside other values (old)
X = preprocessing.scale(X)
X_lately = X[-forecast_out:]
X = X[:-forecast_out]



df.dropna(inplace=True)
Y = np.array(df['label'])


X_train, X_test, Y_train, Y_test = cross_validation.train_test_split(X,Y,test_size =0.2)

#clf1 = svm.SVR(kernel ='poly')
clf = LinearRegression(n_jobs=-1)
#n-jobs is for multithreading

#clf.fit(X_train, Y_train)
clf.fit(X_train, Y_train)
#after training we wannt to store our classifier
#so we use pickling

with open('lin_reg.pickle','wb') as f:
    pickle.dump(clf,f)

pickle_in = open('lin_reg.pickle','rb')
clf = pickle.load(pickle_in)


accuracy = clf.score(X_test, Y_test)
#print(accuracy)
#accuray is squared of error

forecast_set = clf.predict(X_lately)

print(forecast_set, accuracy, forecast_out)
df['Forecast'] = np.nan


last_date = df.iloc[-1].name
last_unix = time.mktime(last_date.timetuple())
one_day = 86400
next_unix = last_unix + one_day

for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += one_day
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)] + [i]

print(df.tail())

df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc = 4)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()


