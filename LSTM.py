# tutorial from https://machinelearningmastery.com/time-series-prediction-lstm-recurrent-neural-networks-python-keras/

import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
np.random.seed(42)

# import dataset
data_csv = sys.argv[1]
category = sys.argv[2]
df=pd.read_csv(data_csv)
data = df['Hours_Spent'].values
# reshape to 2D array
data = data.reshape(-1,1)

# LSTMs are sensitive to the scale of the input data
# normalize the dataset using the MinMaxScaler preprocessing class
# scale data
scaler = MinMaxScaler(feature_range=(0,1))
data = scaler.fit_transform(data)

# split data into training and test sets
train_size = int(len(data) * 0.67)
test_size = len(data) - train_size
train, test = data[0:train_size,:], data[train_size:len(data),:]
train, test = data[0:train_size,:], data[train_size:len(data),:]

# create new dataset
def create_dataset(data, num_steps=1):
    dataX, dataY = [], []
    for i in range(len(data)-num_steps-1):
        a = data[i:(i+num_steps),0]
        dataX.append(a)
        dataY.append(data[i + num_steps, 0])
    return np.array(dataX), np.array(dataY)

num_steps=1
trainX, trainY = create_dataset(train, num_steps)
testX, testY = create_dataset(test, num_steps)

# LSTM network expects the input data (X) to be provided with a specific
# array structure in the form of: [samples, time steps, features]
# reshape data to be [samples, time steps, features]
trainX = np.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = np.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# network has a visible layer with 1 input, a hidden layer with 4 LSTM
# blocks or neurons, and an output layer that makes a single value prediction

# create and fit LSTM network
model = Sequential()
model.add(LSTM(4, input_shape=(1, num_steps)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(trainX, trainY, epochs=100, batch_size=1, verbose=2)

# get predictions
trainPredict = model.predict(trainX)
testPredict = model.predict(testX)

# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])

# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:,0]))
print('Train Score: %.2f RMSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:,0]))
print('Test Score: %.2f RMSE' % (testScore))

#shift train predictions for plotting
trainPredictPlot = np.empty_like(data)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[num_steps:len(trainPredict)+num_steps, :] = trainPredict

# shift test predictions for plotting
testPredictPlot = np.empty_like(data)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(trainPredict)+(num_steps*2)+1:len(data)-1, :] = testPredict

# plot baseline and predictions
# plot baseline and predictions
params = {'figure.figsize': [10,10],'axes.grid': False,'axes.grid.axis': 'both', 'axes.labelsize': 'Medium', 'font.size': 12.0, \
'lines.linewidth': 2}
plt.rcParams.update(params)
plt.plot(scaler.inverse_transform(data), label='actual')
plt.plot(trainPredictPlot, linestyle='--',  label='predicted')
plt.plot(testPredictPlot, linestyle='--', label='predicted')
plt.ylabel('Hours per Week')
plt.title(category)
plt.legend()
plt.show()
