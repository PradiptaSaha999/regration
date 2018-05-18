# -*- coding: utf-8 -*-
"""
Created on Wed May  2 01:40:57 2018

@author: Pradipta
"""

import quandl, math
import numpy as np
import pandas as pd
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression
import datetime
import matplotlib.pyplot as plt
from matplotlib import style
import pickle

style.use('ggplot')

df = quandl.get("WIKI/GOOGL")
df = df[['Adj. Open',  'Adj. High',  'Adj. Low',  'Adj. Close', 'Adj. Volume']]
df['HL_PCT'] = (df['Adj. High'] - df['Adj. Low']) / df['Adj. Close'] * 100.0
df['PCT_change'] = (df['Adj. Close'] - df['Adj. Open']) / df['Adj. Open'] * 100.0

df = df[['Adj. Close', 'HL_PCT', 'PCT_change', 'Adj. Volume']]
forecast_col = 'Adj. Close'
df.fillna(value=-99999, inplace=True)
forecast_out = int(math.ceil(0.01 * len(df)))##0.01*lenth of df so its a number like 30 or 25
df['label'] = df[forecast_col].shift(-forecast_out)##shifiting forecast_col or df['Adj. Close'] so that we can get the future value
#df.dropna(inplace=True)##droping the rows which has no value in any column. We are doing this as we have shifted some coloumns
X = np.array(df.drop(['label'], 1))## Taking all the coloum of df except df['label']
X = preprocessing.scale(X)#normalizing 
X = X[:-forecast_out]##values without forecast_out
X_lately = X[-forecast_out:]##values of forecast_out
df.dropna(inplace=True)##droping the rows which has no value in any column. We are doing this as we have shifted some coloumns

y = np.array(df['label'])
print(len(df['label']))
print(len(y))
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)


clf = LinearRegression()
#clf = LinearRegression(n_jobs=-1)#to use all available thread
clf.fit(X_train, y_train)


with open('linearregression.pickle','wb') as f:#try pickle (wb for write)  
    pickle.dump(clf, f)
    
pickle_in = open('linearregression.pickle','rb')#read from file (rb for read)
clf = pickle.load(pickle_in)


confidence = clf.score(X_test, y_test)
forecast_set = clf.predict(X_lately)
df['Forecast'] = np.nan
last_date = df.iloc[-1].name##taking the last date value 
last_unix = last_date.timestamp()
one_day = 86400
next_unix = last_unix + one_day
for i in forecast_set:
    next_date = datetime.datetime.fromtimestamp(next_unix)
    next_unix += 86400
    df.loc[next_date] = [np.nan for _ in range(len(df.columns)-1)]+[i]
df['Adj. Close'].plot()
df['Forecast'].plot()
plt.legend(loc=10)
plt.xlabel('Date')
plt.ylabel('Price')
plt.show()
print(forecast_set,confidence,forecast_out)
#now try others also
for k in ['linear','poly','rbf','sigmoid']:
    clf = svm.SVR(kernel=k)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    forecast_set = clf.predict(X_lately)
    print(k,forecast_set,confidence,forecast_out)