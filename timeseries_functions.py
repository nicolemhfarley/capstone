import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from statsmodels.graphics import tsaplots
import statsmodels.api as sm

# plt.rcParams.keys()
params = {'figure.figsize': [8,8],'axes.grid.axis': 'both', 'axes.labelsize': 'Medium', 'font.size': 12.0, \
'lines.linewidth': 2}

def index_to_datetime(series):
    "Converts series object indext to datetime"
    series.index = pd.to_datetime(series.index, errors='coerce')

def downsample_data_week(data, fill_method='bfill'):
    downsampled = data.resample(rule='W').mean()
    downsampled.fillna(method=fill_method, inplace=True)
    return downsampled

def plot_series(series, xlabel='', ylabel='', plot_name=''):
    "Plots simple time series from Pandas Series"
    ax = series.plot(figsize=(8,3), linewidth = 3, fontsize=10, grid=True, rot=30)
    ax.set_title(plot_name, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    plt.show()

def plot_series_and_differences(series, ax, num_diff, params, title=''):
    "Plot raw data and specified number of differences"
    plt.rcParams.update(params)
    ax[0].plot(series.index, series)
    ax[0].set_title('Raw series: {}'.format(title))
    for i in range(1, num_diff+1):
        diff = series.diff(i)
        ax[i].plot(series.index, diff)
        ax[i].set_title('Difference # {}'.format(str(i)))

def run_augmented_Dickey_Fuller_test(series, num_diffs=None):
    "Test for stationarity on raw data and specified number of differences."
    test = sm.tsa.stattools.adfuller(series)
    if test[1] >= 0.05:
        print('The p-value for the series is: {p}, which is not significant'.\
        format(p=test[1]))
    else:
        print('The p-value for the series is: {p}, which is significant'.\
        format(p=test[1]))
    if num_diffs:
        for i in range(1, num_diffs +1):
            test = sm.tsa.stattools.adfuller(series.diff(i)[i:])
            if test[1] >= 0.05:
                print('The p-value for difference {diff} is: {p}, which is not \\
                 significant'.format(diff=str(i), p=test[1]))
            else:
                print('The p-value for difference {diff} is: {p}, which is \\
                 significant'.format(diff=str(i), p=test[1]))

def plot_autocorrelation(series, params, lags, alpha=0.05, title=''):
    plt.rcParams.update(params)
    acf_plot = tsaplots.plot_acf(series, lags=lags, alpha=alpha)
    plt.title(title)
    plt.xlabel('Number of Lags')
    plt.show()

def plot_partial_autocorrelation(series, params, lags, alpha=0.05, title=''):
    plt.rcParams.update(params)
    acf_plot = tsaplots.plot_pacf(series, lags=lags, alpha=alpha)
    plt.xlabel('Number of Lags')
    plt.title(title)
    plt.show()

def plot_decomposition(series, params, freq, title=''):
    "Plots observed, trend, seasonal, residual"
    plt.rcParams.update(params)
    decomp = sm.tsa.seasonal_decompose(series, freq=freq)
    fig = decomp.plot()
    plt.title(title)
    plt.show()
