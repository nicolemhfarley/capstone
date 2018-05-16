import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pyflux as pf
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
np.random.seed(42)

###  data = pandas Series


def create_dataset(data, num_steps=1):
    dataX, dataY = [], []
    for i in range(len(data)-num_steps-1):
        a = data[i:(i+num_steps),0]
        dataX.append(a)
        dataY.append(data[i + num_steps, 0])
    return np.array(dataX), np.array(dataY)

def split_and_reshape_data(data, split_at=0.67, num_steps=1):
    train_size = int(len(data) * split_at)
    test_size = len(data) - train_size
    train, test = data[0:train_size,:], data[train_size:len(data),:]
    train, test = data[0:train_size,:], data[train_size:len(data),:]
    trainX, trainY = create_dataset(train, num_steps)
    testX, testY = create_dataset(test, num_steps)
    trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
    testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
    return trainX, trainY, testX, testY

def fit_sequential_LSTM(trainX, trainY, add_layers=4, input_shape=(1,1),\
                        density=1, epochs=100, batch_size=1, optimizer='adam', verbose=2, \
                        loss='mean_squared_error'):
    model = Sequential()
    model.add(LSTM(add_layers, input_shape=input_shape))
    model.add(Dense(density))
    model.compile(loss=loss, optimizer=optimizer)
    model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, verbose=verbose)

def get_LSTM_predictions(trainX, testX):
    "Get predictions for training and test data"
    trainPredict = model.predict(trainX)
    testPredict = model.predict(testX)
    return trainPredict, testPredict

def inverse_transform(trainY, testY, trainPredict, testPredict):
    "Inverse transform train and test set pedictions"
    trainPredict = scaler.inverse_transform(trainPredict)
    trainY = scaler.inverse_transform([trainY])
    testPredict = scaler.inverse_transform(testPredict)
    testY = scaler.inverse_transform([testY])
    return trainY, testY, trainPredict, testPredict

def calculate_RMSE(trainY, testY, trainPredict, testPredict):
    "calculate root mean squared error for training and test predictions"
    trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))

def prep_predictions_for_plotting(data, trainPredict, testPredict, num_steps=1):
    trainPredictPlot = np.empty_like(data)
    trainPredictPlot[:, :] = np.nan
    trainPredictPlot[num_steps:len(trainPredict)+num_steps, :] = trainPredict
    testPredictPlot = np.empty_like(data)
    testPredictPlot[:, :] = np.nan
    testPredictPlot[len(trainPredict)+(num_steps*2)+1:len(data)-1, :] = testPredict
    return trainPredictPlot, testPredictPlot

def plot_data_LSTM_predictions(data, trainPredictPlot, testPredictPlot,\
                               params,title='', xlabel='', ylabel=''):
    fig = plt.figure()
    plt.rcParams.update(params)
    plt.plot(scaler.inverse_transform(data), label='actual')
    plt.plot(trainPredictPlot, linestyle='--',  label='predicted')
    plt.plot(testPredictPlot, linestyle='--', label='predicted')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.show()
    return
