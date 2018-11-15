import copy
import numpy as np

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, Dense

from matplotlib import pyplot as plt


class LSTMNET(Sequential):
    def __init__(self, data, maxlag, lstm_neurons, lstm_cells=1,
                 lossfunc='mean_squared_error', optimizer='adam',
                 epochs=100, batch_size=1, train_size=0.8, verbose=True):

        assert len(data.shape) == 1 or data.shape[1] == 1
        assert maxlag >= 1
        assert 0 < train_size < 1

        self.data = copy.deepcopy(data)
        self.indices = self.data.index
        self.maxlag = maxlag
        self.train_size = int(train_size * len(self.data))
        self.batch_size = batch_size
        self.lstm_neurons = lstm_neurons
        self.lstm_cells = lstm_cells
        self.lossfunc = lossfunc
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose

        self.X_train, self.X_test, self.y_train, self.y_test = self.__stage()

        # run parent-class' constructor
        super().__init__()

    def train(self):
        self.X_train, self.X_test = self.__transform_shape()

        # add LSTM cells
        for _ in range(self.lstm_cells):
            self.add(LSTM(self.lstm_neurons, input_shape=(1, self.maxlag)))
        # add final layer
        self.add(Dense(1))
        self.compile(loss=self.lossfunc,
                     optimizer=self.optimizer)
        self.fit(self.X_train, self.y_train,
                 epochs=self.epochs,
                 batch_size=self.batch_size,
                 verbose=self.verbose)

    def __stage(self):
        """Prepare data for training."""

        self.target = self.data.columns
        cols = ["lag_" + str(i) for i in range(1, self.maxlag+1)]

        # Create lagged variables
        for i, colname in enumerate(cols, 1):
            self.data[colname] = self.data[self.target].shift(-i)

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

    def __inverse_transform_shape(self, X, shape):
        """Transform data into tabular shape."""
        return X.reshape(shape)

    # TODO: finish visualization of fit.
    def plot_fit(self):
        """Visualize fit."""
        train_fit = self.predict(self.X_train)
        test_fit = self.predict(self.X_test)

        # train_fit = self.scaler.inverse_transform(train_fit)
        # test_fit = self.scaler.inverse_transform(test_fit)

        plot_real = np.concatenate((self.y_train, self.y_test))
        plot_train = np.empty_like(plot_real)
        plot_train.fill(np.nan)
        plot_test = np.empty_like(plot_real)
        plot_test.fill(np.nan)

        plot_train[:self.train_size] = train_fit.reshape((-1, ))
        plot_test[self.train_size:] = test_fit.reshape((-1, ))

        idx = np.arange(0, len(self.data)+1, len(self.data) // 1000)
        labels = self.indices

        fig, ax = plt.subplots(1, 1, figsize=(11, 11))
        ax.plot(plot_real, label="data", c="r")
        ax.plot(plot_train, label="train-set", alpha=0.5)
        ax.plot(plot_test, label="test-set", c="g", alpha=0.5)

        plt.xticks(idx, labels, rotation=20)
        plt.title("Crime-Rate fitted by LSTM")
        plt.ylabel("Incidents")
        plt.legend()

        return fig, ax
