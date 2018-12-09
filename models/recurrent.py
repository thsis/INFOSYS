import os
import copy
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential
from keras.layers import LSTM, SimpleRNN, GRU, Dense, Activation

from matplotlib import pyplot as plt


class Recurrent(Sequential):
    """
    Fit recurrent models.

    Convenient class for creating recurrent architectures with different cells.

    * Parameters
        * `data`: pandas.DataFrame with **one** column. The index needs to
           be either a `pandas.DatetimeIndex` or a `pandas.MultiIndex`
           where the first level is a `pandas.DatetimeIndex` and the second
           level a group indicator.
        * `maxlag`: (`int`) maximum number of lags to create from `data`.
        * `cell`: celltype in `[LSTM, GRU, SimpleRNN]` from `keras.layers`.
        * `cell_neurons`: (`int`) neurons in `cell`'s hidden layer.
        * `lossfunc`: loss-function to be used.
        * `optimizer`: optimizer to be used.
        * `cellkwargs`: (`dict`) `kwargs` to `keras.Sequential.fit`-method.
        * `fitkwargs`: (`dict`) keyword arguments for `cell.`
        * `epochs`: (`int`) number of training-epochs.
        * `batch_size`: (`int`) number of samples per batch.
        * `train_size`: (`int`) propotion of training samples.
          0 < train_size <= 1.
        * `verbose`: (`bool`) level of verbosity during training.
    """

    def __init__(self, data, maxlag, cell, cell_neurons=4, num_cells=1,
                 lossfunc='mean_squared_error', optimizer='adam',
                 cellkwargs={}, fitkwargs={}, epochs=10, batch_size=1,
                 train_size=0.8, verbose=True):
        """Instantiate Recurrent class."""

        assert maxlag >= 1
        assert 0 < train_size <= 1
        assert len(data.columns) == 1

        # Operate solely on a copy to avoid implicitly changing the data
        self.data = copy.deepcopy(data.sort_index())
        self.target = self.data.columns
        self.maxlag = maxlag

        # Parse Data: assume first index is the date, second the cross section.
        try:
            self.time_label = self.data.index.names[0]
            self.cross_label = self.data.index.names[1]
            self.cross_dimension = self.data.index.unique(self.cross_label)
        except IndexError:
            self.time_label = self.data.index.name
            self.cross_label = None
            self.cross_dimension = [None]

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
        self.fitkwargs = fitkwargs
        self.cellkwargs = cellkwargs

        self.X_train, self.X_test, self.y_train, self.y_test = self.__stage()

        # run parent-class' constructor
        super().__init__()

    def train(self):
        """Train the model based on parameters passed to `__init__`."""
        X_train = self.__transform_shape(self.X_train)

        self.add(self.cell(self.cell_neurons,
                           input_shape=(1, self.maxlag),
                           **self.cellkwargs))
        self.add(Dense(1))
        self.add(Activation('relu'))

        self.compile(loss=self.lossfunc,
                     optimizer=self.optimizer)
        self.fit(X_train, self.y_train,
                 epochs=self.epochs,
                 batch_size=self.batch_size,
                 verbose=self.verbose,
                 **self.fitkwargs)

    def forecast(self, X):
        """
        Calculate predictions.

        * Parameters:
            * `X`: `numpy.ndarray` of shape (`n`, `maxlag`), `n` is arbitrary.

        * Returns:
            * `out`: predictions of model of shape (`n`, 1).
        """
        if X.shape[1] == self.maxlag:
            X_ = self.__transform_shape(X)
            return self.predict(X_)

        else:
            X_, _ = self.__get_features(X)
            return self.predict(X_.values)

    def __stage(self):
        """
        Prepare data for training.

        * Returns:
           * `(X_train, X_test, y_train, y_test)`: `numpy.ndarrays`
        """

        # Separate train and test data
        locate_train, locate_test = self.__locate_train_test(self.data)
        train = self.data.loc[locate_train]

        # Scale: fit scaler only on **training** data.
        self.scaler = MinMaxScaler()
        self.scaler.fit(train.values)

        # Scale the **whole** series
        self.data.loc[:, "y"] = self.scaler.transform(self.data.values)

        # Create lagged variables and split into target and features.
        X, y = self.__get_features()

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

    def __get_features(self):
        """
        Create lagged variables in `self.data`.

        * Returns:
            * (X, y): `numpy.ndarray`s of feature and target values.
        """
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

        return X, y

    def __transform_shape(self, X):
        """
        Transform data into a feedable form for `cell`-layer.

        * Parameters:
            * `X`: `numpy.ndarray` of shape `(num_samples, maxlag)`

        * Returns:
            * `out`: `numpy.ndarray` of shape (num_samples, 1, maxlag)
        """
        N, M = X.shape
        new_shape = (N, 1, M)
        out = X.reshape(new_shape)

        return out

    def __inverse_transform_shape(self, X):
        """
        Transform data into tabular shape.

        * Parameters:
            * `X`: `numpy.ndarray` of shape `(num_samples, 1, maxlag)`

        * Returns:
            * `out`: `numpy.ndarray` of shape `(num_samples, maxlag)`
        """
        N, _, M = X.shape
        old_shape = (N, M)
        out = X.reshape(old_shape)

        return out

    def __locate_train_test(self, data):
        """
        Create logical vector for indexing.

        * Parameters:
            * `data`: `pandas.DataFrame` with a `pandas.DatetimeIndex`.

        * Returns:
            * `is_train`: `boolean` array. `True` if row belongs to train set.
            * `ìs_test`: `boolean` array. `True` if row belongs to test set.
        """
        try:
            # Works if MultiIndex, breaks if not.
            is_test = data.index.map(lambda x: x[0] > self.split_at)
            is_train = data.index.map(lambda x: x[0] <= self.split_at)
        except TypeError:
            # Fall back to DatetimeIndex.
            is_test = data.index.map(lambda x: x > self.split_at)
            is_train = data.index.map(lambda x: x <= self.split_at)
        return is_train, is_test

    def plot_fit(self, savepath=None, **kwargs):
        """
        Visualize fit of training and test data.

        * Parameters:
            * `savepath`: Filename to save figure, not saved if `None`.
            * `**kwargs`: Keyword arguments to `pyplot.subplots`.

        * Returns:
            * `(fig, ax)`: Figure and axis, depends on index of `self.data`.
                + if `pandas.DatetimeIndex`: single plot of whole series.
                + if `pandas.MultiIndex`: multiple subplots for each series.
        """
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
            fig, ax = self.__plot_multi_series(**kwargs)
        else:
            fig, ax = self.__plot_single_series(**kwargs)

        if savepath:
            plt.savefig(savepath)

        return fig, ax

    def __plot_single_series(self, **kwargs):
        """
        Plot train- and test-fit for single series.

        * Parameters:
            * `**kwargs`: keyword arguments to `pyplot.subplots`
        * Returns:
            * `(fig, ax)`: figure and axis of single plot.
        """
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
                c="orange", label="Test-Fit", alpha=0.8)

        plt.title("Crime-Rate fitted by {}".format(self.cell.__name__))
        plt.ylabel("Incidents")
        plt.legend()

        return fig, ax

    def __plot_multi_series(self, **kwargs):
        """
        Plot train- and test-fit for multiple series.

        * Parameters:
            * `**kwargs`: keyword arguments to `pyplot.subplots`
        * Returns:
            * `(fig, ax)`: figure and axis of multiple subplots.
        """
        fig, axes = plt.subplots(len(self.cross_dimension), 1, sharex=True,
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
                    c="orange", label="Test-Fit", alpha=0.8)

        fig.text(0.04, 0.5, 'Incidents', va='center', rotation='vertical')
        fig.suptitle("Crime-Rate fitted by {}".format(self.cell.__name__))
        plt.legend()

        return fig, axes


if __name__ == "__main__":
    print("Generate show-cases.")
    # Series by district.
    print("Grouped by district:")
    datapath = os.path.join("data", "crimes_district.csv")
    dataset = pd.read_csv(datapath, index_col=["Date", "District"],
                          dtype={"crimes_district_total": np.float32},
                          parse_dates=["Date"])
    lstm = Recurrent(dataset, 2, cell=LSTM, epochs=1)
    lstm.train()
    lstm.plot_fit(os.path.join("models", "lstm-district-show-case.png"),
                  figsize=(10, 20))

    rnn = Recurrent(dataset, 2, cell=SimpleRNN, epochs=1)
    rnn.train()
    rnn.plot_fit(os.path.join("models", "rnn-district-show-case.png"),
                 figsize=(10, 20))

    gru = Recurrent(dataset, 2, cell=GRU, epochs=1)
    gru.train()
    gru.plot_fit(os.path.join("models", "gru-district-show-case.png"),
                 figsize=(10, 20))

    # Total series.
    print("Total series:")
    datapath = os.path.join("data", "crime_total.csv")
    dataset = pd.read_csv(datapath, index_col=["date"],
                          dtype={"crimes_total": np.float32},
                          parse_dates=["date"])
    lstm = Recurrent(dataset, 2, cell=LSTM, epochs=1)
    lstm.train()
    lstm.plot_fit(os.path.join("models", "lstm-total-show-case.png"),
                  figsize=(20, 10))

    rnn = Recurrent(dataset, 2, cell=SimpleRNN, epochs=1)
    rnn.train()
    rnn.plot_fit(os.path.join("models", "rnn-total-show-case.png"),
                 figsize=(20, 10))

    gru = Recurrent(dataset, 2, cell=GRU, epochs=1)
    gru.train()
    gru.plot_fit(os.path.join("models", "gru-total-show-case.png"),
                 figsize=(20, 10))
