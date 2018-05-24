import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.arima_process import ArmaProcess

###  data = pandas Series
# params = {'figure.figsize': [8,8],'axes.grid.axis': 'both', 'axes.grid': True, 'axes.labelsize': 'Medium', 'font.size': 12.0, \
# 'lines.linewidth': 2}
# plt.rcParams.update(params)


def get_AR_model(data, order):
    """Fits AR model
    Inputs:
        data: pandas Series
        order: (p,d) format
    Oututs:
        returns values from fitted model
    """
    model = ARMA(data, order=order)
    results = model.fit()
    return results

def plot_AR_model(data, order, start, end, title='', xlabel='', ylabel=''):
    """Plots AR model forcast against true data
    Inputs:
        data: pandas Series
        order: (p,d) format (int)
        start/end: starting/ending dates for plot (x_axis)
        title/xlabel/ylabel: labels for plot
    """
    results = get_AR_model(data, order)
    results.plot_predict(start=start, end=end)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()

# check goodness of fit for a range of parameters for AR model
def get_AR_model_order_BIC(data, max_order_plus_one):
    """Calculates Baysian Information Criterion for range of model orders
    Inputs:
        data: pandas Series
        max_order_plus_one: (int) max p value +1 to test
    Outputs:
        BIC_array: np array of BIC values for each value of p
    """
    BIC_array = np.zeros(max_order_plus_one)
    for p in range(1, max_order_plus_one):
        results = get_AR_model(data, order=(p,0))
        BIC_array[p] = results.bic
    return BIC_array

def plot_BIC_AR_model(data, max_order_plus_one):
    """Plots BIC for range of orders
    Inputs:
        data: pandas Series
        max_order_plus_one: (int) max p value +1 to test
    Outputs:
        graph of BIC values (y-axis) vs p values (x-axis)
    """
    array = get_AR_model_order_BIC(data, max_order_plus_one)
    plt.plot(range(1, max_order_plus_one), array[1:max_order_plus_one], marker='o')
    plt.xlabel('Order of {mod} Model'.format(mod='AR'))
    plt.ylabel('Baysian Information Criterion')
    plt.show()

def get_MA_model(data, order):
    """Fits MA model
    Inputs:
        data: pandas Series
        order: (d,q) format (int)
    Oututs:
        returns values from fitted model
    """
    model = ARMA(data, order=order)
    results = model.fit()
    return results

def plot_MA_model(data, order, start, end, title='', xlabel='', ylabel=''):
    """Plots MA model forcast against true data
    Inputs:
        data: pandas Series
        order: (p,d) format (int)
        start/end: starting/ending dates for plot (x_axis)
        title/xlabel/ylabel: labels for plot
    Outputs:
        graph of MA model
    """
    results = get_MA_model(data, order)
    results.plot_predict(start=start, end=end)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.show()

### Do these below functions actually work?  Not yet tested
# check goodness of fit for a range of parameters for MA model?
def get_MA_model_order_BIC(data, max_order_plus_one):
    """Calculates Baysian Information Criterion for range of model orders
    Inputs:
        data: pandas Series
        max_order_plus_one: (int) max q value +1 to test
    Outputs:
        BIC_array: np array of BIC values for each value of q
    """
    "Calculates Baysian Information Criterion for range of model orders"
    BIC_array = np.zeros(max_order_plus_one)
    for q in range(1, max_order_plus_one):
        results = get_MA_model(data, order=(0,q))
        BIC_array[q] = results.bic
    return BIC_array

def plot_BIC_MA_model(data, max_order_plus_one):
    """Plots BIC for range of orders
    Inputs:
        data: pandas Series
        max_order_plus_one: (int) max q value +1 to test
    Outputs:
        graph of BIC values (y-axis) vs q values (x-axis)
    """    array = get_MA_model_order_BIC(data, max_order_plus_one)
    plt.plot(range(1, max_order_plus_one), array[1:max_order_plus_one], marker='o')
    plt.xlabel('Order of {mod} Model'.format(mod='ARMA'))
    plt.ylabel('Baysian Information Criterion')
    plt.show()

def get_MA_train_test_predictions(training_data, test_data, order, start, end):
    """ Get MA predictions
    Inputs:
        training and test data: pandas Series
        order: (d,q) format
        start/end: (str) starting/ending dates
    Outputs:
        data_plus_forecast: dataframe with original data and forecast plot_all_df_columns
        forecast: just predictions
    """
    training_data = training_data.to_frame()
    test_data = test_data.to_frame()
    results = ARMA(training_data, order=order).fit()
    forecast = results.predict(start=start, end=end).to_frame()
    all_data = pd.concat([training_data, test_data], axis=0)
    data_plus_forecast = pd.merge(left=all_data, right=forecast, how='outer', left_index=True, right_index=True)
    data_plus_forecast.columns = ['data', 'forecast']
    return forecast, data_plus_forecast

def get_MA_train_test_MSE(df, data_col, pred_col, train_end, test_start, data_name=''):
    """ Get MA MSE for training and test data
    Inputs:
        df: pandas dataframe of original data and ARIMAX prediction to be split into both train and test sets
        data_col = (str) name of df column containing original data
        pred_col = (str) name of df column containing model predictions
        train_end/test_start: (str) ending date for training set and starting data for test set
        data_name: (str) for labeling output
    Outputs:
        data_plus_forecast: dataframe with original data and forecast plot_all_df_columns
        forecast: just predictions
    """
    train_error_df = df.loc[:train_end]
    test_error_df = df.loc[test_start:]
    for col in train_error_df.columns:
        train_error_df = train_error_df[train_error_df[col].notnull()]
    mse_train = mean_squared_error(train_error_df[data_col], train_error_df[pred_col])
    mse_test = mean_squared_error(test_error_df[data_col], test_error_df[pred_col])
    print('train MSE: {}'.format(mse_train))
    print('test MSE: {}'.format(mse_test))
    return mse_train, mse_test
