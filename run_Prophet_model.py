import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from fbprophet import Prophet

# import data from csv files for a specific provider cateogry

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
    provider['Hours'] = provider['AppointmentDuration'] / 60.0
    # return provider series
    return provider

def get_provider_weekly_hours(provider):
    provider_hours = provider['Hours']
    index_to_datetime(provider)
    provider = provider.resample('W-MON').sum()
    provider_hours = provider[1:]
    provider_hours = provider_hours['Hours']
    return provider_hours

def get_number_unique_providers(provider):
    num_provider = provider['Provider'].resample('W-MON', lambda x: x.nunique())
    # set index to to_datetime
    index_to_datetime(num_provider)
    # drop incomplete first column
    num_provider = num_provider[1:]
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

def get_holidays():
    # make dataframe for each holiday
    christmas_dates = ['2015-12-25', '2016-12-25', '2017-12-25']
    new_year_dates = ['2016-01-01', '2017-01-01', '2018-01-01']
    thanksgiving_dates = ['2015-11-26', '2016-11-24', '2017-11-23']
    thanksgiving = pd.DataFrame({'holiday':'Thanksgiving', 'ds': pd.to_datetime(thanksgiving_dates)})
    christmas = pd.DataFrame({'holiday':'Christams', 'ds': pd.to_datetime(christmas_dates)})
    new_years = pd.DataFrame({'holiday':'New Years', 'ds': pd.to_datetime(new_year_dates)})
    # combine into single holidays DataFrame
    holidays = pd.concat([christmas, thanksgiving, new_years])
    return holidays

def get_prophet_forecast_w_holidays(df, date_hours_cols,\
        pred_cols, periods, holidays, mean_hours_provider):
    """
    Inputs:
        df: dataframe containing timeseries and weekly hours
        date_hours_cols: (list) names for columns containing the date and weekly hours data
        pred_cols: (list) name of columns containing estimated hours, upper and lower limits
        of estimates
        periods: (int) number of periods to forecast.
        holidays: (dataframe) of holidays with holiday names, dates (datetime
            format, upper and lower windows (ints, optional))
    Outputs:
        Prophet model
        forecast
        df with original data plus predictions and upper/lower predictions
    """
    df['ds'] = df.index
    df['y'] = df['Hours']
    df = df[['ds', 'y']]
    model = Prophet(holidays=holidays)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    forecast.index = forecast['ds']
    df_pred = pd.concat([df, forecast[pred_cols]], axis=1)
    # make num providers column
    df_pred['Predicted_num_Providers'] = round(df_pred['yhat'] / mean_hours_provider, 1)
    predictions = forecast.iloc[-periods:]
    return model, forecast, df_pred

def run_Prophet_model(df_cols):
        # import provider data
    provider = get_cleaned_provider_data(csv_file, category_name)
        # get weekly hours data
    provider_hours = get_provider_weekly_hours(provider)
        # get number of providers data
    num_provider = get_number_unique_providers(provider)
        # merge provider dataframes
    providers = merge_hours_and_providers(provider_hours, num_provider)
        # get hours per provider
    provider_df, avg_provider_hours = get_hours_per_provider(providers)
    holidays = get_holidays()
        # get prophet model, forecast, predictions dataframe
    model, forecast, df_pred = get_prophet_forecast_w_holidays(provider_df,\
            date_hours_cols, pred_cols, periods, holidays, avg_provider_hours)
    df_pred.columns = df_cols
    df_pred.to_csv(outfile)

if __name__ == '__main__':
    csv_file = './data/appointments_through_04-2018.csv'
    df_cols = ['date', 'Hours', 'Predicted Hours', 'Lower Limit', 'Upper Limit',\
    'Predicted Number Providers']
    date_hours_cols = ['date', 'Hours']
    category_name = 'RN/PA'
    periods = 90
    pred_cols = ['yhat', 'yhat_lower', 'yhat_upper']
    outfile = './data/RNPA_prophet_forecast.csv'

    run_Prophet_model(df_cols)
