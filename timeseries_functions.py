import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
from statsmodels.graphics import tsaplots
import statsmodels.api as sm

params = {'figure.figsize': [8,8],'axes.labelsize': 'Medium', 'font.size': 12.0, \
'lines.linewidth': 2}

def plot_series_and_differences(series, ax, num_diff, title):
    "Plot raw data and specified number of differences"
    ax[0].plot(series.index, series)
    ax[0].set_title('Raw series: {}'.format(title))
    for i in range(1, num_diff+1):
        diff = series.diff(i)
        ax[i].plot(series.index, diff)
        ax[i].set_title('Difference # {}'.format(str(i)))

def run_augmented_Dickey_Fuller_test(series, num_diffs=None):
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

def plot_autocorrelation(series, params, lags, alpha, title):
    plt.rcParams.update(params)
    acf_plot = tsaplots.plot_acf(series, lags=lags, alpha=alpha)
    plt.title(title)
    plt.xlabel('Number of Lags')
    plt.show()

def plot_partial_autocorrelation(series, params, lags, alpha, title):
    plt.rcParams.update(params)
    acf_plot = tsaplots.plot_pacf(series, lags=lags, alpha=alpha)
    plt.xlabel('Number of Lags')
    plt.title(title)
    plt.show()

def plot_decomposition(series, params, freq, title):
    "Plots observed, trend, seasonal, residual"
    plt.rcParams.update(params)
    decomp = sm.tsa.seasonal_decompose(series, freq=freq)
    fig = decomp.plot()
    plt.title(title)
    plt.show()
