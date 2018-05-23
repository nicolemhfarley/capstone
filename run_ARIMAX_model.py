# import sys
import numpy as np
import pandas as pd
from statsmodels.graphics import tsaplots
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA, ARIMAResults
from sklearn.metrics import mean_squared_error


def index_to_datetime(series):
    """Converts pandas dataframe or series index to datetime"""
    series.index = pd.to_datetime(series.index, errors='coerce')

# import data from csv files for a specific provider cateogry
def get_cleaned_provider_data(csv_file, category_name):
    appointment_df = pd.read_csv(csv_file, index_col=0)
    # convert index to datetime if necessary
    if type(appointment_df.index) != True:
        index_to_datetime(appointment_df)
    # group by specialty
    provider = appointment_df[appointment_df['Specialty'] == category_name]
    # convert appointment duration into hours
    provider['Hours'] = provider['AppointmentDuration'] / 60
    # return provider series
    return provider

def get_provider_weekly_hours(provider):
    index_to_datetime(provider)
    provider = provider.resample('W-MON').sum()
    provider_hours = provider[1:]
    provider_hours = provider_hours['Hours']
    print(provider_hours)
    return provider_hours

def get_number_unique_providers(provider):
    num_provider = provider['Provider'].resample('W-MON', lambda x: x.nunique())
    # set index to to_datetime
    index_to_datetime(num_provider)
    # drop incomplete first column
    num_provider = num_provider[1:]
    print(num_provider)
    return num_provider

def merge_hours_and_providers(hours, num_providers):
    hours = hours.to_frame()
    num_providers = num_providers.to_frame()
    df = pd.merge(left=num_providers, right=hours, how='inner', left_index=True, right_index=True)
    return df

def get_hours_per_provider(df):
    df.columns = ['Number_Providers', 'Hours']
    df['Hours_per_Provider'] = df['Hours'] / df['Number_Providers']
    mean_hours_provider = df['Hours_per_Provider'].mean()
    return df, mean_hours_provider

def get_ARIMAX_predictions(data, order, start, end, exog=None, typ='levels'):
    """Get ARIMAX predictions
    Inputs:
        data: pandas Series
        order: (p,d,q) format
        start/end: (str) starting/ending dates
        exog: data for exogenous variable as pandas series
    Outputs:
        data_plus_forecast: dataframe with original data and forecast plot_all_df_columns
    """
    data = data.to_frame()
    results = ARIMA(data, order=order, exog=exog).fit()
    forecast = results.predict(start=start, end=end, exog=exog, typ=typ).to_frame()
    data_plus_forecast = pd.merge(left=data, right=forecast, how='outer', left_index=True, right_index=True)
    data_plus_forecast.columns = ['data', 'forecast']
    return data_plus_forecast

def get_ARIMAX_forecast(csv_file, category_name, order, start_date, end_date, outfile):
    # import provider data
    provider = get_cleaned_provider_data(csv_file, category_name)
    # get weekly hours data
    provider_hours = get_provider_weekly_hours(provider)
    # get number of providers data
    num_provider = get_number_unique_providers(provider)
    # merge provider dataframes
    provider = merge_hours_and_providers(provider_hours, num_provider)
    # get hours per provider
    provider_df, avg_provider_hours = get_hours_per_provider(provider)
    # get forecast df
    forecast_df = get_ARIMAX_predictions(data=provider_hours, order=order, start=start_date,\
     end=end_date, exog=num_provider, typ='levels')
    # rename columns
    forecast_df.columns = ['Hours', 'Predicted_Hours']
    # get predicted number of providers rounded up
    forecast_df['Predicted_Num_Providers'] = round((forecast_df['Predicted_Hours'] / avg_provider_hours),2)
    # get forecast
    forecast = forecast_df[start_date:end_date]#[['Predicted_Hours', 'Predicted_Num_Providers']]
    # keep only date in index, drop time
    forecast.index = forecast.index.date
    # output to csv file
    forecast.to_csv(outfile)

if __name__ == '__main__':
    infile = './data/appointments_through_04-2018.csv'
    category_name = 'doctors'
    order = (5,1,3)
    start_date = '2015-01-19'
    end_date = '2018-09-30'
    csv_out = 'test_arimax.csv'
    # get predictions data
    get_ARIMAX_forecast(csv_file=infile, category_name='doctor', order=(5,1,3),\
    start_date='2015-01-19', end_date='2018-09-30', outfile='test_arimax_pred.csv')
