# -*- coding: utf-8 -*-
"""
Created on Mon May 21 17:15:14 2018

@author: Pradipta
"""

import regrationFromScrach as rg
import numpy as np
import random
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


def create_dataset(hm,variance,step=2,correlation=False):
    val = 1
    ys = []
    for i in range(hm):
        y = val + random.randrange(-variance,variance)
        ys.append(y)
        if correlation and correlation == 'pos':
            val+=step
        elif correlation and correlation == 'neg':
            val-=step

    xs = [i for i in range(len(ys))]
    
    return np.array(xs, dtype=np.float64),np.array(ys,dtype=np.float64)


if __name__ == '__main__':
    xs, ys = create_dataset(40,40,2,correlation='pos')
    m = rg.best_fit_slope(xs,ys)
    print(m)
    
    
    c=rg.best_fit_c(xs,ys,m)
    print(c)
    
    
    regression_line = rg.regression_line(m,c,xs)##   corrosponding y accroding to x
    
                       
    r_squared = rg.coefficient_of_determination(ys,regression_line)##  R-squired error
    print(r_squared)                   
    
    
    predict_x = 7
    predict_y = (m*predict_x)+c
    print(predict_y)
    
    
    plt.scatter(xs,ys,color='#003F72',label='data')
    plt.scatter(predict_x,predict_y,color='#0FFF72',label='new Value')
    plt.plot(xs, regression_line, label='regression line')
    plt.legend(loc=4)
    plt.show()
