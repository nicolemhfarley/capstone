import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA, ARMA
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.stats.diagnostic import acorr_ljungbox
from scipy import signal

def get_ARIMA_model(data, order):
    "Fits ARIMA model"
    arima = ARIMA(data, order=order)
    results = arima.fit()
    summary = results.summary()
    params = results.params
    residuals = results.resid
    return results, summary, params, residuals

def plot_ARIMA_model(data, order, start, end, title='', xlabel='', ylabel=''):
    "Plots ARIMA model"
    results = ARIMA(data, order=order).fit()
    fig = results.plot_predict(start=start, end=end)
    plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.show()

def plot_ARIMA_resids(data, order, start, end, title='', xlabel='', ylabel=''):
    "Plots ARIMA model residuals"
    results = ARIMA(data, order=order).fit().resid
    residuals.plot(figsize=(5,5))
    plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.show()

def get_ARIMA_forecast(data, order, start, end, typ=None):
    "Predicts future values of time series"
    results = ARIMA(data, order=order).fit()
    forecast = results.predict(start=start, end=end, typ=typ)
    return forecast

def plot_data_plus_ARIMA_predictions(data, order, start, end, typ=None, figsize=(10,10), title='', ylabel='', xlabel=''):
    "Make forecast and plot as extension of the existing time series data"
    forecast = get_ARIMA_forecast(data, order, start, end, typ=typ)
    data_plus_forecast = pd.concat([data, forecast], axis=1)
    data_plus_forecast.columns = ['data', 'predicted']
    data_plus_forecast.plot(figsize=(12,8), grid=True)
    plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.show()

def plot_data_plus_ARIMA_predictions(data, order, start, end, typ=None, figsize=(10,10), title='', ylabel='', xlabel=''):
    results = ARIMA(data, order=order).fit()
    forecast = results.predict(start=start, end=end, typ=typ)
    data_plus_forecast = pd.concat([data, forecast], axis=1)
    data_plus_forecast.columns = ['data', 'forecast']
    data_plus_forecast.plot(figsize=(12,8))
    plt.title(title)
    plt.ylabel(xlabel)
    plt.xlabel(ylabel)
    plt.show()

def test_rolling_ARIMA_forecast(train_data, test_data, order):
    "Calculates rolling ARIMA forecast, returns predicted vs actual"
    history = [x for x in train]
    predictions = []
    for t in range(len(test)):
        arima = ARIMA(history, order=order)
        arima_fitted = arima.fit()
        forecast = arima_fitted.forecast()
        yhat = forecast[0]
        predictions.append(yhat)
        observed = test[t]
        history.append(observed)
    return predictions, test

def plot_rolling_ARIMA_forecast(train_data, test_data, order, title):
    "Calculates and plots rolling ARIMA forecast"
    predicted, expected = test_rolling_ARIMA_forecast(train_data, test_data, order)
    predictions = np.hstack(predicted)
    df = pd.DataFrame({'predicted': predictions, 'actual':test})
    df.plot()
    plt.title(title)
    plt.show()

def get_predictions_df_and_plot_rolling_ARIMA_forecast(train_data, test_data, order, title):
    "Calculates and plots rolling ARIMA forecast"
    predicted, expected = test_rolling_ARIMA_forecast(train_data, test_data, order)
    predictions = np.hstack(predicted)
    actual = pd.concat([train_data, test_data], axis=0 )
    df = pd.DataFrame({'predicted': predictions, 'actual':expected})
    real_and_predicted_df = pd.DataFrame({'actual': actual, 'predicted':df['predicted']})
    real_and_predicted_df.plot()
    plt.title(title)
    plt.show()
    return df
