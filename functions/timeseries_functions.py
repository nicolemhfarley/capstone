import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics import tsaplots
import statsmodels.api as sm


###  data = pandas Series

#plt.rcParams.keys()
params = {'figure.figsize': [8,8],'axes.grid.axis': 'both', 'axes.grid': True, 'axes.labelsize': 'Medium', 'font.size': 12.0, \
'lines.linewidth': 2}
# plt.rcParams.update(params)

def index_to_datetime(series):
    """Converts pandas dataframe or series index to datetime"""
    series.index = pd.to_datetime(series.index, errors='coerce')

def weekly_resample(data):
    """resamples hours data to weekly and sums column values
    """
    data = data.resample('W-MON').sum()
    return data

def plot_all_df_columns(df, col_nums, title='', xlabel=''):
    """Plots the data in each column of a dataframe as subplots.
    Inputs:
        df: pandas DataFrame
        col_nums: column index in integers
        title/xlabel/ylabel: (str) labels for plot
    """
    i = 1
    values = df.values
    for col in col_nums:
        plt.subplot(len(col_nums), 1, i)
        plt.plot(values[:, col])
        plt.title(title)
        plt.ylabel(dr_df.columns[col])
        plt.xlabel(xlabel)
        i += 1
    plt.tight_layout()
    plt.show()

def plot_series(series, figsize=None, xlabel='', ylabel='', plot_name='',\
                v_lines=None):
    """Plots simple time series from Pandas Series
    Inputs:
        series: name of pandas Series
        figsize: size of figure (x,y) dimensions
        plot_name/xlabel/ylabel: (str) labels for plot
        v_lines: location of vertical lines on the x axis
    """
    ax = series.plot(figsize=figsize, linewidth = 3, fontsize=10, grid=True, rot=30)
    ax.set_title(plot_name, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ymin,ymax= ax.get_ylim()
    ax.vlines(x=v_lines, ymin=ymin, ymax=ymax-1, color='red', linestyle='--')
    plt.show()

def plot_series_save_fig(series, figsize=(10,10), xlabel='', ylabel='', plot_name='',\
                v_lines=None, figname=None):
    """Plots simple timeseries and saves to specified file
    Inputs:
        series: name of pandas Series
        figsize: size of figure (x,y) dimensions
        plot_name/xlabel/ylabel: (str) labels for plot
        v_lines: location of vertical lines on the x axis
        figname: (str) name for saving figure including file extension
    """
    ax = series.plot(figsize=figsize, linewidth = 3, fontsize=10, grid=True, rot=30)
    ax.set_title(plot_name, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=15)
    ax.set_ylabel(ylabel, fontsize=15)
    ymin,ymax= ax.get_ylim()
    ax.vlines(x=v_lines, ymin=ymin, ymax=ymax-1, color='red', linestyle='--')
    plt.savefig(figname)
    plt.show()

def plot_series_and_differences(series, ax, num_diff, title=''):
    """Plot raw timeseries data and specified number of differences
    Inputs:
        series: name of pandas Series
        ax: plt.axes
        num_diffs: (int) the number of series differences that should be ploted
        title: str) title for plot
    """
    plt.xticks(rotation=40)
    ax[0].plot(series.index, series)
    ax[0].set_title('Raw series: {}'.format(title))
    ax[0].set_xticklabels(labels=series.index.date, rotation=45)
    for i in range(1, num_diff+1):
        diff = series.diff(i)
        ax[i].plot(series.index, diff)
        ax[i].set_title('Difference # {}'.format(str(i)))
        ax[i].set_xticklabels(labels=series.index.date, rotation=45)

def run_augmented_Dickey_Fuller_test(series, num_diffs=None):
    """Test for stationarity on raw timeseries data and specified number of differences
    using augmented Dickey-Fuller test
    Inputs:
        series: name of pandas Series
        num_diffs: (int) the number of series differences to be tested
    Outputs:
        ADF statistic for each difference
        1/5/10%: critical values for each difference
        p-value for each difference
    """
    test = sm.tsa.stattools.adfuller(series)
    print('ADF Statistic: {}'.format(test[0]))
    print('Critical values:')
    for key, value in test[4].items():
        print('{k}: {v}'.format(k=key, v=value))
    if test[1] >= 0.05:
        print('The p-value for the series is: {p}'.format(p=test[1]))
    else:
        print('The p-value for the series is: {p}'.format(p=test[1]))
    if num_diffs:
        for i in range(1, num_diffs +1):
            test = sm.tsa.stattools.adfuller(series.diff(i)[i:])
            print('ADF Statistic: {}'.format(test[0]))
            print('Critical values:')
            for key, value in test[4].items():
                print('{k}: {v}'.format(k=key, v=value))
            if test[1] >= 0.05:
                print('The p-value for difference {diff} is: {p}'.format(diff=str(i), p=test[1]))
            else:
                print('The p-value for difference {diff} is: {p}'.format(diff=str(i), p=test[1]))

def plot_autocorrelation(series, params, lags, alpha=0.05, title=''):
    """Plots autocorrelation of timeseries
    Inputs:
        series: name of pandas Series
        params: plt.rcParams
        lags: (int) number of lags to include
        alpha: significant level
        title: (str) title for plot
    """
    plt.rcParams.update(params)
    acf_plot = tsaplots.plot_acf(series, lags=lags, alpha=alpha)
    plt.title(title)
    plt.xlabel('Number of Lags')
    plt.show()

def plot_partial_autocorrelation(series, params, lags, alpha=0.05, title=''):
    """Plots partial autocorrelation of timeseries w/ datetime index
    Inputs:
        series: name of pandas Series
        params: plt.rcParams
        lags: (int) number of lags to include
        alpha: significant level
        title: (str) title for plot
    """
    plt.rcParams.update(params)
    acf_plot = tsaplots.plot_pacf(series, lags=lags, alpha=alpha)
    plt.xlabel('Number of Lags')
    plt.title(title)
    plt.show()

def get_seasonal_decomposition(series, freq=None):
    """Seasonally decomposes timeseries
    Inputs:
        series: pandas Series
        freq: (int) frequency of the series
    Outputs:
        returns seasonal decomposition components
    """
    decomp = sm.tsa.seasonal_decompose(series, freq=freq)
    return decomp.seasonal, decomp.trend, decomp.resid

def plot_decomposition(series, params, freq=None, title=''):
    """Plots observed, trend, seasonal, residuals of timeseries
    Inputs:
        series: name of pandas Series
        params: plt.rcParams
        freq: (int) frequency of the series
        title: (str) title for plot
    """
    plt.rcParams.update(params)
    decomp = sm.tsa.seasonal_decompose(series, freq=freq)
    fig = decomp.plot()
    plt.title(title)
    plt.show()

def plot_2_series_double_yaxis(x, y1, y2, figsize=(10,10), fontsize=12, title='', \
                               y1_label='', y2_label='', xlabel='', savefig=False,\
                               figname='figure'):
    """Plots 2 time series sharing same x x_axis
    Inputs:
        x: x values
        y1: values for left y-axis
        y2: values for right y-axis
        figsize: size of figure (x,y)
        title/x_label/y1_label/y2_label: (str) labels for plot
        savefig: (bool) whether to save figure to file
        figname: (str) name of figure w/ file extension
    Outputs:
        figure
        file of saved figure
    """
    x = x
    y1 = y1
    y2 = y2
    fig, ax = plt.subplots(figsize=figsize, sharex=True)
    ax2 = ax.twinx()
    ax.set_title(title, fontsize=fontsize+4)
    ax.plot(x, y1, 'r-', label=y1_label)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(y1_label, fontsize=fontsize)
    ax.set_xticklabels(labels=x, rotation=45)
    ax2.plot(x, y2, 'b-', label=y2_label)
    ax2.set_ylabel(y2_label, fontsize=fontsize)
    ax.legend(loc='upper left')
    ax2.legend(loc='lower right')
    plt.show()
    if savefig == True:
        fig.savefig(figname)
        
## plot detrended data using functions from matt drury tine series lecture w/ some
# modifications

def make_col_vector(array):
    """Convert a one dimensional numpy array to a column vector."""
    return array.reshape(-1, 1)

def make_design_matrix(array):
    """Construct a design matrix from a numpy array, including an intercept term."""
    return sm.add_constant(make_col_vector(array), prepend=False)

def fit_linear_trend(series):
    """Fit a linear trend to a time series.  Return the fit trend as a numpy array."""
    X = make_design_matrix(np.arange(len(series)) + 1)
    linear_trend_ols = sm.OLS(series.values, X).fit()
    linear_trend = linear_trend_ols.predict(X)
    return linear_trend

def plot_trend_data(ax, series):
    """Plots timeseries
    """
    ax.plot(series.index, series)

def plot_linear_trend(ax, series, title='', xlabel='', ylabel=''):
    """Fits and plots linear trend to timeseries data
    """
    linear_trend = fit_linear_trend(series)
    plot_trend_data(ax, title, series)
    ax.plot(series.index, linear_trend)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
