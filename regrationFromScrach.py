# -*- coding: utf-8 -*-
"""
Created on Fri May 18 18:37:01 2018

@author: Pradipta
"""

from statistics import mean
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')

##   y=mx+c


def best_fit_slope(xs,ys):##    calculating slope m
    m = (((mean(xs)*mean(ys)) - mean(xs*ys)) /
         ((mean(xs)**2) - mean(xs**2)))
    return m

def best_fit_c(xs,ys,m):##  calculating c
    c=( mean(ys) - (m*(mean(xs))))
    return c

def squared_error(ys_orig,ys_line): ##   this is just Squired error not R-squired it is the squire of the error
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))

def coefficient_of_determination(ys_orig,ys_line):##    R-squired error
    y_mean_line = [mean(ys_orig) for y in ys_orig]##    set all value to mean valu to make ybar line
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)
def regression_line(m,c,xs):
    regression_line = [(m*x)+c for x in xs]##   corrosponding y accroding to x
    return regression_line


if __name__ == '__main__':
    xs = np.array([1,2,3,4,5,6], dtype=np.float64)
    ys = np.array([5,4,6,5,6,7], dtype=np.float64)
    m = best_fit_slope(xs,ys)
    print(m)
    
    
    c=best_fit_c(xs,ys,m)
    print(c)
    
    
    regression_line = regression_line(m,c,xs)##   corrosponding y accroding to x
    
                       
    r_squared = coefficient_of_determination(ys,regression_line)##  R-squired error
    print(r_squared)                   
    
    
    predict_x = 7
    predict_y = (m*predict_x)+c
    print(predict_y)
    
    
    plt.scatter(xs,ys,color='#003F72',label='data')
    plt.scatter(predict_x,predict_y,color='#0FFF72',label='new Value')
    plt.plot(xs, regression_line, label='regression line')
    plt.legend(loc=4)
    plt.show()