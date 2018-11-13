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


class Feeder():
    """Prepares data to be fed into an LSTM model."""

    def __init__(self):
        pass

    def fit_transform(self, X, new_shape, y=None):
        """Reshapes X, new_shape should be (time, observations, dimensions)"""

        return X.reshape(new_shape)


class LSTMNET(Sequential):
    def __init__(self, X_train, y_train, lstm_neurons, input_shape=(1, 1),
                 lstm_cells=1, lossfunc='mean_squared_error', optimizer='adam',
                 epochs=100, batch_size=1):
        # run parent-class' constructor
        super().__init__()
        # add LSTM cells
        for _ in range(lstm_cells):
            self.add(LSTM(lstm_neurons, input_shape=input_shape))
        # add final layer
        self.add(Dense(1))
        self.compile(loss=lossfunc,
                     optimizer=optimizer)
        self.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def __prepare(self):
        """Prepare data for training."""



X_train, X_test, y_train, y_test = prepare_data(dataset, maxlag=3)

feeder = Feeder()
X_train = feeder.fit_transform(X_train, (X_train.shape[0], 1, 3))

model = LSTMNET(X_train, y_train, lstm_neurons=4, input_shape=(1, 3))


def objective(params):
    model = LSTMNET(X_train, y_train, **params)
