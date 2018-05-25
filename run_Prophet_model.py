import sys
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from fbprophet import Prophet


##### unfinished



# import data from csv files for a specific provider cateogry
# def get_provider_data(csv_file):
#     provider_df = pd.read_csv(csv_file)
#     return provider_df

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
    provider_hours = provider.copy()
    provider_hours = provider.groupby(provider.index.date)['Hours'].sum()
    index_to_datetime(provider)
    provider = provider.resample('W-MON').sum()
    provider_hours = provider[1:]
    provider_hours = provider_hours['Hours']
    return provider_hours

def get_number_unique_providers(provider):
    num_provider = provider.copy()
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

def get_prophet_forecast_w_holidays(df, df_cols, date_hours_cols,\
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
    df.columns = df_cols
    df = df[date_hours_cols]
    df.columns = ['ds', 'y']
    model = Prophet(holidays=holidays)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    df_pred = pd.concat([df, forecast[pred_cols]], axis=1)
    # make num providers column
    df_pred['Predicted_num_Providers'] = round(df_pred['yhat'] / mean_hours_provider)
    # predictions = forecast.iloc[-periods:]
    # get_prophet_training_mse(df_pred, df_name, periods)
    # get_prophet_test_mse(df_pred, df_name, periods)
    return model, forecast, df_pred

def prophet_forecast_to_csv(df_pred, file_name):
    """Save prophet predictions in dataframe format to csv file"""
    prediction_df.columns = ['Date', 'True_Hours', 'Predicted_Hours', 'Lower_Limit',\
     'Upper_Limit', 'Predicted_num_Providers']
    prediction_df.to_csv('{}_predictions.csv'.format(file_name))

def run_prophet_forecast(csv_file, category_name, date_hours_cols,\
        pred_cols, periods, holidays, mean_hours_provider, out_csv):
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
    holidays = get_holidays()
    # get prophet model, forecast, predictions dataframe
    model, forecast, df_pred = get_prophet_forecast_w_holidays(provider_df,\
        df_cols, date_hours_cols, pred_cols, periods, holidays)
    prophet_forecast_to_csv(df_pred, outfile)



if __name__ == '__main__':
    infile = './data/appointments_through_04-2018.csv'
    df_cols = ['date', 'Number_Providers', 'Hours', 'Hours_per_Provider']
    date_hours_cols = ['date', 'Hours']
    periods = 90
    pred_cols = ['yhat', 'yhat_lower', 'yhat_upper']
    # start_date = '2015-01-12'
    # end_date = '2018-09-30'
    outfile = 'dr_test_prophet_forecast.csv'
    run_prophet_forecast(csv_file=infile, category_name='doctors', date_hours_cols=date_hours_cols,\
            pred_cols=pred_cols, periods=90, holidays, mean_hours_provider, out_csv=outfile)
