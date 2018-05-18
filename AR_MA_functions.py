import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_process import ArmaProcess

###  data = pandas Series
# plt.rcParams.update(params)
params = {'figure.figsize': [8,8],'axes.grid.axis': 'both', 'axes.grid': True, 'axes.labelsize': 'Medium', 'font.size': 12.0, \
'lines.linewidth': 2}

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
# check goodness of fit for a range of parameters for MA model?
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

def get_MA_train_test_predictions(training_data, test_data, order, start, end):
    training_data = training_data.to_frame()
    test_data = test_data.to_frame()
    results = ARMA(training_data, order=order).fit()
    forecast = results.predict(start=start, end=end).to_frame()
    all_data = pd.concat([training_data, test_data], axis=0)
    data_plus_forecast = pd.merge(left=all_data, right=forecast, how='outer', left_index=True, right_index=True)
    data_plus_forecast.columns = ['data', 'forecast']
    return forecast, data_plus_forecast

def get_MA_train_test_MSE(df, data_col, pred_col, train_end, test_start, data_name=''):
    train_error_df = df.loc[:train_end]
    test_error_df = df.loc[test_start:]
    for col in train_error_df.columns:
        train_error_df = train_error_df[train_error_df[col].notnull()]
    mse_train = mean_squared_error(train_error_df[data_col], train_error_df[pred_col])
    mse_test = mean_squared_error(test_error_df[data_col], test_error_df[pred_col])
    print('train MSE: {}'.format(mse_train))
    print('test MSE: {}'.format(mse_test))
    return mse_train, mse_test
