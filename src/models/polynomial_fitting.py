import numpy as np
from scipy.optimize import curve_fit
from typing import Callable
import pandas as pd

def get_R_squared(y:np.array,y_hat:np.array) -> float:
    '''
    Returns the $R^2$ for real and predicted variables.

    :param y: real y values
    :param y_hat: predicted y values
    :return R_squared: Calculated as $1 - \fract{SS_residuals}{SS_total}$
    '''
    y_mean = np.mean(y)
    ssr = np.sum((y-y_hat)**2)
    sst = np.sum((y-y_mean)**2)
    R_squared = 1 - (ssr/sst)
    return R_squared

def quadratic(x,a,b,c):
    '''
    Describing a polynomial function of the form of $y = a*x^2 + b*x + c$
    '''
    return a * x**2 + b * x + c

def fit_polynomial_for_group(df_grouped:pd.DataFrame, group:list[int], polynomial_function: Callable):
    '''
    Fits a polynomial function for values corresponding to indices `group` in `df_grouped`, then randomly chooses

    :param group: Group of indices having similar paths
    :param df_grouped: DataFrame grouped by `track_id``
    :param polynomial_function: A polynomial function to be fitted using the supplied data
    :return params: Fitted parameters for the polynomial function
    :return covariance: TBD
    '''

    x = np.array([x 
        for i in group
        for x in np.array(df_grouped['x'][i])])
    y = np.array([y 
        for i in group
        for y in np.array(df_grouped['y'][i])])
    
    params, covariance = curve_fit(polynomial_function, x, y)
    

    return (params,covariance)

