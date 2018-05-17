import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error
from fbprophet import Prophet

def get_prophet_training_mse(forecast, df_name, periods):
    """compute error over all known dates, actual vs yhat"""
    predictions = forecast.iloc[0:-periods]
    mse = mean_squared_error(predictions['y'], predictions['yhat'])
    print('MSE for {name} training set is {error}'.format(name=df_name, error=mse))

def get_prophet_test_mse(forecast, df_name, periods):
    """compute error over all known dates, actual vs yhat"""
    predictions = forecast.iloc[-152:-periods]
    predictions.dropna(inplace=True, axis=0)
    mse = mean_squared_error(predictions['y'], predictions['yhat'])
    print('MSE for {name} test set is {error}'.format(name=df_name, error=mse))

def get_prophet_forecast(df, df_name, df_cols, date_hours_cols, pred_cols, periods):
    """
    """
    df.columns = df_cols
    df = df[date_hours_cols]
    df.columns = ['ds', 'y']
    model = Prophet()
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    df_pred = pd.concat([df, forecast[pred_cols]], axis=1)
    predictions = forecast.iloc[-periods:]
    get_training_mse(df_pred, df_name, periods)
    get_test_mse(df_pred, df_name, periods)
    return model, forecast, df_pred

def get_prophet_forecast_w_holidays(df, df_name, df_cols, date_hours_cols, \
                pred_cols, periods, holidays=holidays):
    df.columns = df_cols
    df = df[date_hours_cols]
    df.columns = ['ds', 'y']
    model = Prophet(holidays=holidays)
    model.fit(df)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    df_pred = pd.concat([df, forecast[pred_cols]], axis=1)
    predictions = forecast.iloc[-periods:]
    get_prophet_training_mse(df_pred, df_name, periods)
    get_prophet_test_mse(df_pred, df_name, periods)
    return model, forecast, df_pred

def plot_prophet_forecast(model, forecast, xlabel='', ylabel=''):
    """Plots forecast and confidence interval from prophet model
    input: model (Prophet Object), forecast: predictions, df_name: name
    to use to label y-axis.
    """
    model.plot(forecast, xlabel=xlabel, ylabel=ylabel)
    model.plot_components(forecast)

def prophet_forecast_to_csv(prediction_df, file_name):
    """Save prophet predictions to csv file"""
    prediction_df.columns = ['Date', 'True_Hours', 'Predicted_Hours', 'Lower_Limit', 'Upper_Limit']
    prediction_df.to_csv('{}_predictions.csv'.format(file_name))
