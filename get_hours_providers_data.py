import numpy as np
import pandas as pd

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
    print(provider)
    return provider

def get_provider_weekly_hours(provider):
    index_to_datetime(provider)
    provider = provider.resample('W-MON').sum()
    provider_hours = provider[1:]
    provider_hours = provider_hours['Hours']
    print(provider_hours)
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

def get_hours_providers_df(csv_file, category_name, outfile_name):
    provider = get_cleaned_provider_data(csv_file, category_name)
    provider_hours = get_provider_weekly_hours(provider)
    num_provider = get_number_unique_providers(provider)
    print(num_provider)
    df = merge_hours_and_providers(provider_hours, num_provider)
    # print(df.head())
    df.to_csv(outfile_name)

if __name__ == '__main__':
    infile = './data/appointments_through_04-2018.csv'
    get_hours_providers_df(csv_file=infile, category_name='doctors', outfile_name='test_get_data.csv')
