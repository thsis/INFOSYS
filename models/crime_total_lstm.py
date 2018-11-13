import os
import copy
import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from keras.models import Sequential
from keras.layers import LSTM, Dense

from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from hyperopt.pyll.base import scope


datapath = os.path.join("data", "crime_total.csv")
dataset = pd.read_csv(datapath, index_col="date",
                      dtype={"crimes_total": np.float32})


class LSTMNET(Sequential):
    def __init__(self, data, maxlag, lstm_neurons, lstm_cells=1,
                 lossfunc='mean_squared_error', optimizer='adam',
                 epochs=100, batch_size=1, train_size=0.8):

        assert len(data.shape) == 1 or data.shape[1] == 1
        assert maxlag >= 1
        assert 0 < train_size < 1

        self.data = copy.deepcopy(data)
        self.maxlag = maxlag
        self.train_size = int(train_size * len(self.data))
        self.batch_size = batch_size
        self.lstm_neurons = lstm_neurons
        self.lstm_cells = lstm_cells
        self.lossfunc = lossfunc
        self.optimizer = optimizer
        self.epochs = epochs

        self.X_train, self.X_test, self.y_train, self.y_test = self.__stage()

    def train(self):
        self.X_train, self.X_test = self.__transform_shape()

        # run parent-class' constructor
        super().__init__()
        # add LSTM cells
        for _ in range(self.lstm_cells):
            self.add(LSTM(self.lstm_neurons, input_shape=(1, self.maxlag)))
        # add final layer
        self.add(Dense(1))
        self.compile(loss=self.lossfunc,
                     optimizer=self.optimizer)
        self.fit(self.X_train, self.y_train,
                 epochs=self.epochs, batch_size=self.batch_size)

        # self.X_train, self.X_test = self.__inverse_transform_shape()

    def __stage(self):
        """Prepare data for training."""

        target = self.data.columns
        cols = ["lag_" + str(i) for i in range(1, self.maxlag+1)]

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

        self.old_train_shape = X_train.shape
        self.old_test_shape = X_test.shape

        return X_train, X_test, y_train, y_test

    def __transform_shape(self):
        """Transform data into a feedable form for LSTM-layer."""
        # Infer new shape
        self.new_train_shape = (self.old_train_shape[0],
                                1,
                                self.maxlag)
        self.new_test_shape = (self.old_test_shape[0],
                               1,
                               self.maxlag)

        X_train = self.X_train.reshape(self.new_train_shape)
        X_test = self.X_test.reshape(self.new_test_shape)

        return X_train, X_test

    def __inverse_transform_shape(self):
        """Transform data into tabular shape."""
        X_train = self.X_train.reshape(self.old_train_shape)
        X_test = self.X_test.reshape(self.old_test_shape)

        return X_train, X_test

    # TODO: finish visualization of fit.
    def plot_fit(self):
        """Visualize fit."""
        pass


model = LSTMNET(dataset, maxlag=3, lstm_neurons=4, lstm_cells=1,
                epochs=10, batch_size=3)
model.train()


def objective(params):
    model = LSTMNET(dataset, **params)
    model.train()
    predictions = model.predict(model.X_test)
    loss = mean_squared_error(y_true=model.y_test, y_pred=predictions)
    return {'loss': loss, 'status': STATUS_OK}


paramspace = {

}
