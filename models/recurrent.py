import os
import copy
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, GRU, Dense

from matplotlib import pyplot as plt


class Recurrent(Sequential):
    def __init__(self, data, maxlag, cell=LSTM, cell_neurons=4, num_cells=1,
                 lossfunc='mean_squared_error', optimizer='adam',
                 epochs=100, batch_size=1, train_size=0.8, verbose=True):

        assert maxlag >= 1
        assert 0 < train_size < 1

        # Operate solely on a copy to avoid implicitly changing the data
        self.data = copy.deepcopy(data.sort_index())

        # Parse Data: assume first index is the date, second the cross section.
        try:
            self.time_label = self.data.index.names[0]
            self.cross_label = self.data.index.names[1]
            self.cross_dimension = self.data.index.unique(self.cross_label)
        except IndexError:
            self.time_label = self.data.index.name
            self.cross_label = None
            self.cross_dimension = [None]

        # TODO: check if we still need those
        self.target = self.data.columns
        self.indices = self.data.index
        self.maxlag = maxlag

        # Obtain train/test-size.
        self.train_size = train_size
        self.time_dimension = self.data.index.unique(self.time_label)
        self.split_at_idx = int(self.train_size * len(self.time_dimension))
        self.split_at = self.time_dimension[self.split_at_idx]

        # Save parameters for LSTM-architecture
        self.cell = cell
        self.batch_size = batch_size
        self.num_cells = num_cells
        self.lossfunc = lossfunc
        self.optimizer = optimizer
        self.epochs = epochs
        self.verbose = verbose
        self.cell_neurons = cell_neurons

        self.X_train, self.X_test, self.y_train, self.y_test = self.__stage()

        # run parent-class' constructor
        super().__init__()

    def train(self):
        X_train = self.__transform_shape(self.X_train)

        # add LSTM cells
        for _ in range(self.num_cells):
            self.add(self.cell(self.cell_neurons,
                               input_shape=(self.batch_size, self.maxlag)))
        # add final layer
        self.add(Dense(1))
        self.compile(loss=self.lossfunc,
                     optimizer=self.optimizer)
        self.fit(X_train, self.y_train,
                 epochs=self.epochs,
                 batch_size=self.batch_size,
                 verbose=self.verbose)

    def __stage(self):
        """Prepare data for training."""

        # Separate train and test data
        locate_train, locate_test = self.__locate_train_test(self.data)
        train = self.data.loc[locate_train]

        # Scale
        # Fit scaler only on **training** data.
        self.scaler = MinMaxScaler()
        self.scaler.fit(train.values)

        # Scale the **whole** series
        self.data.loc[:, "y"] = self.scaler.transform(self.data.values)

        # Create lagged variables.
        cols = ["lag_" + str(i) for i in range(1, self.maxlag+1)]

        for i, colname in enumerate(cols, 1):
            if isinstance(self.data.index, pd.core.index.MultiIndex):
                lag = self.data.groupby(self.cross_label)["y"].shift(i)
            else:
                lag = self.data["y"].shift(i)

            self.data[colname] = lag

        self.data = self.data.dropna(axis=0)
        # Split into target and features
        y = self.data.iloc[:, 1]
        X = self.data.iloc[:, 2:]

        # Split into train- and test-set
        locate_X_train, locate_X_test = self.__locate_train_test(X)
        locate_y_train, locate_y_test = self.__locate_train_test(y)

        X_train = X.loc[locate_X_train]
        X_test = X.loc[locate_X_test]

        y_train = y.loc[locate_y_train]
        y_test = y.loc[locate_y_test]

        # Obtain unique times/cross section identifiers.
        self.train_time = len(X_train.index.unique(self.time_label))
        self.test_time = len(X_test.index.unique(self.time_label))

        if self.cross_label is not None:
            self.train_cross = len(X_train.index.unique(self.cross_label))
            self.test_cross = len(X_test.index.unique(self.cross_label))
        else:
            self.train_cross = 1
            self.test_cross = 1

        return X_train.values, X_test.values, y_train.values, y_test.values

    def __transform_shape(self, ndarray):
        """Transform data into a feedable form for LSTM-layer."""
        N, M = ndarray.shape
        new_shape = (N, 1, M)
        out = ndarray.reshape(new_shape)

        return out

    def __inverse_transform_shape(self, ndarray):
        """Transform data into tabular shape."""
        N, _, M = ndarray.shape
        old_shape = (N, M)
        out = ndarray.reshape(old_shape)

        return out

    def __locate_train_test(self, data):
        try:
            # Works if MultiIndex, breaks if not.
            locate_test = data.index.map(lambda x: x[0] > self.split_at)
            locate_train = data.index.map(lambda x: x[0] <= self.split_at)
        except TypeError:
            # Fall back to DatetimeIndex.
            locate_test = data.index.map(lambda x: x > self.split_at)
            locate_train = data.index.map(lambda x: x <= self.split_at)
        return locate_train, locate_test

    def plot_fit(self, **kwargs):
        """Visualize fit."""
        X_train = self.__transform_shape(self.X_train)
        X_test = self.__transform_shape(self.X_test)

        train_fit = self.predict(X_train)
        test_fit = self.predict(X_test)

        # Rescale
        train_fit = self.scaler.inverse_transform(train_fit)
        test_fit = self.scaler.inverse_transform(test_fit)

        self.data["predictions"] = np.concatenate([train_fit, test_fit])
        self.data["status"] = np.concatenate([
            np.repeat("train", len(self.y_train)),
            np.repeat("test", len(self.y_test))])

        if isinstance(self.data.index, pd.core.index.MultiIndex):
            fig, ax = self.__plot_multi_series()
        else:
            fig, ax = self.__plot_single_series()

        return fig, ax

    def __plot_single_series(self, **kwargs):
        is_train = self.data.status == "train"
        train = self.data.loc[is_train, "predictions"]
        test = self.data.loc[~is_train, "predictions"]
        truth = self.data.loc[:, self.target]

        fig, ax = plt.subplots(1, 1, **kwargs)

        ax.plot(truth.index, truth.values,
                c="red", label="Ground Truth")
        ax.plot(train.index, train.values,
                c="blue", label="Train-Fit", alpha=0.8)
        ax.plot(test.index, test.values,
                c="g", label="Test-Fit", alpha=0.8)

        plt.title("Crime-Rate fitted by {}".format(self.cell.__name__))
        plt.ylabel("Incidents")
        plt.legend()

        return fig, ax

    def __plot_multi_series(self, **kwargs):
        fig, axes = plt.subplots(len(self.cross_dimension), 1,
                                 **kwargs)
        for cr, ax in zip(self.cross_dimension, axes):
            is_train = self.data.status == "train"
            is_cr = self.data.index.map(lambda x: x[1] == cr).values
            train = self.data.loc[is_train & is_cr, "predictions"]
            test = self.data.loc[~is_train & is_cr, "predictions"]
            truth = self.data.loc[is_cr, self.target]

            ax.plot(truth.index.droplevel(self.cross_label),
                    truth.values,
                    c="red", label="Ground Truth")
            ax.plot(train.index.droplevel(self.cross_label),
                    train.values,
                    c="blue", label="Train-Fit", alpha=0.8)
            ax.plot(test.index.droplevel(self.cross_label),
                    test.values,
                    c="g", label="Test-Fit", alpha=0.8)

        plt.suptitle("Crime-Rate fitted by {}".format(self.cell.__name__))
        plt.ylabel("Incidents")
        plt.legend()

        return fig, axes


if __name__ == "__main__":
    datapath = os.path.join("data", "crimes_district.csv")
    dataset = pd.read_csv(datapath, index_col=["Date", "District"],
                          dtype={"crimes_district_total": np.float32},
                          parse_dates=["Date"])
    lstm = Recurrent(dataset, 2, cell=LSTM, epochs=1)
    lstm.train()
    lstm.plot_fit()

    rnn = Recurrent(dataset, 2, cell=SimpleRNN, epochs=1)
    rnn.train()
    rnn.plot_fit()

    gru = Recurrent(dataset, 2, cell=GRU, epochs=1)
    gru.train()
    gru.plot_fit()


if __name__ == "__main__":
    datapath = os.path.join("data", "crime_total.csv")
    dataset = pd.read_csv(datapath, index_col=["date"],
                          dtype={"crimes_total": np.float32},
                          parse_dates=["date"])
    lstm = Recurrent(dataset, 2, cell=LSTM, epochs=1)
    lstm.train()
    lstm.plot_fit()

    rnn = Recurrent(dataset, 2, cell=SimpleRNN, epochs=1)
    rnn.train()
    rnn.plot_fit()

    gru = Recurrent(dataset, 2, cell=GRU, epochs=1)
    gru.train()
    gru.plot_fit()
