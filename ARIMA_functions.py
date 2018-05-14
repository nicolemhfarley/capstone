import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults

###  data = pandas Series

params = {'figure.figsize': [8,8],'axes.grid.axis': 'both', 'axes.grid': True, 'axes.labelsize': 'Medium', 'font.size': 12.0, \
'lines.linewidth': 2}

def get_ARIMA_model(data, order):
    """Fits ARIMA model
    data: pandas Series
    order: (p,d,q) format
    """
    arima = ARIMA(data, order=order)
    results = arima.fit()
    summary = results.summary()
    params = results.params
    residuals = results.resid
    return results, summary, params, residuals

def plot_ARIMA_model(data, order, start, end, title='', xlabel='', ylabel=''):
    """Plots ARIMA model forcast against true data with confidence interval
    data: pandas Series
    order: (p,d,q) format
    start/end: starting/ending dates for plot (x_axis)
    """
    results = ARIMA(data, order=order).fit()
    fig = results.plot_predict(start=start, end=end)
    plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.show()

def plot_ARIMA_resids(data, order, start, end, title='', xlabel='', ylabel=''):
    """Plots ARIMA model residuals
    data: pandas Series
    order: (p,d,q) format
    start: start date
    end: at the latest = end date in index
    """
    residuals = ARIMAResults(data, order=order).fit().resid
    residuals.plot(figsize=(5,5))
    plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.show()

def get_ARIMAX_model(data, order, exog_var):
    """Fits ARIMAX model
    data: pandas Series
    order: (p,d,q) format
    exog_var = exogenous variable as pandas Series
    """
    arima = ARIMA(endog=data, order=order, exog=exog_var)
    results = arima.fit()
    summary = results.summary()
    params = results.params
    residuals = results.resid
    return results, summary, params, residuals

def plot_ARIMAX_model(data, order, exog_var, start, end, params=None, title='', xlabel='', ylabel=''):
    """Plots ARIMAX model
    data: pandas Series
    order: (p,d,q) format
    exog_var = exogenous variable as pandas Series
    start/end: starting/ending dates for plot (x_axis)
    """
    plt.rcParams.update(params)
    results = ARIMA(endog=data, order=order, exog=exog_var).fit()
    fig = results.plot_predict(start=start, end=end,exog=exog_var)
    plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.show()

def get_ARIMA_forecast(data, order, start, end, typ=None):
    """Predicts future values of time series
    data: pandas Series
    order: (p,d,q) format
    start/end: starting/ending dates for plot (x_axis)
    """
    results = ARIMA(data, order=order).fit()
    forecast = results.predict(start=start, end=end, typ=typ)
    return forecast

def plot_ARIMA_forecast_and_CI(train_data, test_data, order, start, end, params,\
 alpha=0.05, title=''):
    """
    train/test data as pandas series
    order: (p,d,q) format
    start/end: starting/ending dates for plot (x_axis)
    params: for plt.plot
    alpha: specifies confidence interval (0.05 => 95 percent CI)
    Returns: Plot of data and forecast with confidence inteval
    """
    start=start
    end=end
    fitted_model = ARIMA(train_data, order=order).fit()
    predicted, expected = test_rolling_ARIMA_forecast(train_data, test_data, order=order)
    params = params
    plt.rcParams.update(params)
    fig = fitted_model.plot_predict(start=start, end=end, alpha=alpha)
    plt.show()

def plot_data_plus_ARIMA_predictions(data, order, start, end, typ='levels',\
 figsize=(10,10), title='', ylabel='', xlabel=''):
    """
    data: pandas Series
    order: (p,d,q) format
    start/end: starting/ending dates for plot (x_axis)
    """

    results = ARIMA(data, order=order).fit()
    forecast = results.predict(start=start, end=end, typ=typ)
    data_plus_forecast = pd.concat([data, forecast], axis=1)
    data_plus_forecast.columns = ['data', 'forecast']
    data_plus_forecast.plot(figsize=figsize)
    plt.title(title)
    plt.grid(True)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.show()

def plot_data_plus_ARIMAX_predictions(data, order, start, end, exog=None, typ='levels',\
 figsize=(10,10), title='', ylabel='', xlabel=''):
    results = ARIMA(data, order=order, exog=exog).fit()
    forecast = results.predict(start=start, end=end, exog=exog, typ=typ)
    data_plus_forecast = pd.concat([data, forecast], axis=1)
    data_plus_forecast.columns = ['data', 'forecast']
    data_plus_forecast.plot(figsize=figsize)
    plt.title(title)
    plt.grid(True)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.show()

def test_rolling_ARIMA_forecast(train_data, test_data, order):
    "Calculates rolling ARIMA forecast, returns predicted vs actual"
    history = [x for x in train_data]
    predictions = []
    for t in range(len(test_data)):
        arima = ARIMA(history, order=order)
        arima_fitted = arima.fit()
        forecast = arima_fitted.forecast()
        yhat = forecast[0]
        predictions.append(yhat)
        observed = test_data[t]
        history.append(observed)
    return predictions, test_data

def get_predictions_df_and_plot_rolling_ARIMA_forecast(train_data, test_data, \
order, figsize=(10,5),title=''):
    "Calculates and plots rolling ARIMA forecast"
    fitted_model = ARIMA(train_data, order=order).fit()
    predicted, expected = test_rolling_ARIMA_forecast(train_data, test_data, order)
    predictions = np.hstack(predicted)
    actual = pd.concat([train_data, test_data], axis=0 )
    df = pd.DataFrame({'predicted': predictions, 'actual':expected})
    real_and_predicted_df = pd.DataFrame({'actual': actual, 'predicted':df['predicted']})
    real_and_predicted_df.plot(figsize=figsize)
    plt.title(title)
    plt.show()
    return df
