# -*- coding: utf-8 -*-
"""
Created on Mon May 21 17:33:04 2018

@author: Pradipta
"""

import regrationFromScrach as rg
import quandl, math
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from sklearn import cross_validation
from matplotlib import style
style.use('ggplot')

if __name__ == '__main__':
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
    X = np.array(df.drop(['label', 'HL_PCT', 'PCT_change', 'Adj. Volume'], 1))## removing all the coloum of df except df['Adj. Close']
    X = X[:-forecast_out].flatten()##values without forecast_out
    X_lately = X[-forecast_out:]##values of forecast_out
    df.dropna(inplace=True)##droping the rows which has no value in any column. We are doing this as we have shifted some coloumns
    
    y = np.array(df['label']).flatten()
    print(len(df['label']))
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.2)
    m = rg.best_fit_slope(X_train,y_train)
    print(m)
    
    
    c=rg.best_fit_c(X_train,y_train,m)
    print(c)
    
    
    regression_line = rg.regression_line(m,c,X_test)##   corrosponding y accroding to x
    
                       
    r_squared = rg.coefficient_of_determination(y_test,regression_line)##  R-squired error
    print(r_squared)                   
    
    
    predict_x = 7
    predict_y = (m*predict_x)+c
    print(predict_y)
    
    
    plt.scatter(X_train,y_train,color='#003F72',label='data')
    #plt.scatter(predict_x,predict_y,color='#0FFF72',label='new Value')
    plt.plot(X_train, regression_line, label='regression line')
    plt.legend(loc=4)
    plt.show()
