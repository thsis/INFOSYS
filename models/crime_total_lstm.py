import os
import copy
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import LSTM, Dense

datapath = os.path.join("data", "crime_total.csv")
dataset = pd.read_csv(datapath, index_col="date",
                      dtype={"crimes_total": np.float32})


class LSTMNET(Sequential):
    def __init__(self, data, maxlag, lstm_neurons, input_shape=(1, 1),
                 lstm_cells=1, lossfunc='mean_squared_error', optimizer='adam',
                 epochs=100, batch_size=1, train_size=0.8):

        assert len(data.shape) == 1
        assert maxlag >= 1
        assert 0 < train_size < 1

        self.data = data
        self.maxlag = maxlag
        self.old_shape = data.shape
        self.new_shape = input_shape
        self.train_size = int(train_size * len(self.data))

        self.X_train, self.X_test, self.y_train, self.y_test = self.__stage()
        self.X_train, self.X_test = self.__transform_shape()

        # run parent-class' constructor
        super().__init__()
        # add LSTM cells
        for _ in range(lstm_cells):
            self.add(LSTM(lstm_neurons, input_shape=input_shape))
        # add final layer
        self.add(Dense(1))
        self.compile(loss=lossfunc,
                     optimizer=optimizer)
        self.fit(self.X_train, self.y_train,
                 epochs=epochs, batch_size=batch_size)

    def __stage(self):
        """Prepare data for training."""

        target = self.data.columns
        cols = ["lag_" + str(i) for i in range(1, maxlag+1)]

        # Create lagged variables
        for i, colname in enumerate(cols, 1):
            self.data[colname] = self.data[target].shift(-i)

        self.data = self.data[:-self.maxlag]

        # Scale
        self.scaler = MinMaxScaler()
        self.data = self.scaler.fit_transform(self.data.values)

        # Split into target and features
        y = self.data[:, 0]
        X = self.data[:, 1:]

        # Split into train- and test-set
        y_train, y_test = y[:self.train_size], y[self.train_size:]
        X_train, X_test = X[:self.train_size], X[self.train_size:]

        return X_train, X_test, y_train, y_test

    def __transform_shape(self):
        """Transform data into a feedable form for LSTM-layer."""
        X_train = self.X_train.reshape(self.new_shape)
        X_test = self.X_test.reshape(self.new_shape)

        return X_train, X_test

    def __inverse_transform_shape(self):
        """Transform data into tabular shape."""
        X_train = self.X_train.reshape(self.old_shape)
        X_test = self.X_test.reshape(self.old_shape)

        return X_train, X_test


model = LSTMNET(data, maxlag=3, lstm_neurons=4, input_shape=(1, 3))


def objective(params):
    model = LSTMNET(X_train, y_train, **params)


def prepare_data()
