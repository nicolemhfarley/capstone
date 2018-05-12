import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_process import ArmaProcess

###  data = pandas Series

def get_AR_model(data, order):
    model = ARMA(data, order=order)
    results = model.fit()
    return results

def plot_AR_model(data, order, start, end, title='', xlabel='', ylabel=''):
    results = get_AR_model(data, order)
    results.plot_predict(start=start, end=end)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()

# check goodness of fit for a range of parameters for AR model
def get_AR_model_order_BIC(data, max_order_plus_one):
    "Calculates Baysian Information Criterion for range of model orders"
    BIC_array = np.zeros(max_order_plus_one)
    for p in range(1, max_order_plus_one):
        results = get_AR_model(data, order=(p,0))
        BIC_array[p] = results.bic
    return BIC_array

def plot_BIC_AR_model(data, max_order_plus_one):
    "Plots BIC for range of orders"
    array = get_AR_model_order_BIC(data, max_order_plus_one)
    plt.plot(range(1, max_order_plus_one), array[1:max_order_plus_one], marker='o')
    plt.xlabel('Order of {mod} Model'.format(mod='AR'))
    plt.ylabel('Baysian Information Criterion')
    plt.show()

def get_MA_model(data, order):
    "pd.Series data"
    model = ARMA(data, order=order)
    results = model.fit()
    return results

def plot_MA_model(data, order, start, end, title='', xlabel='', ylabel=''):
    results = get_MA_model(data, order)
    results.plot_predict(start=start, end=end)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()

### Do these below functions actually work?  Not yet tested
# check goodness of fit for a range of parameters for MA model
def get_MA_model_order_BIC(data, max_order_plus_one):
    "Calculates Baysian Information Criterion for range of model orders"
    BIC_array = np.zeros(max_order_plus_one)
    for q in range(1, max_order_plus_one):
        results = get_MA_model(data, order=(0,q))
        BIC_array[q] = results.bic
    return BIC_array

def plot_BIC_MA_model(data, max_order_plus_one):
    "Plots BIC for range of orders"
    array = get_MA_model_order_BIC(data, max_order_plus_one)
    plt.plot(range(1, max_order_plus_one), array[1:max_order_plus_one], marker='o')
    plt.xlabel('Order of {mod} Model'.format(mod='ARMA'))
    plt.ylabel('Baysian Information Criterion')
    plt.show()
