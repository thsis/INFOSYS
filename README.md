# Seminar Information Systems

This repository contains code for illustrating the *Long Short Term Memory* (**LSTM**) and *Gated Recurrent Unit* (**GRU**) architectures for neural networks as well as comparing them to traditional *Recurrent Neural Networks* (**RNN**).

## Crime Data

This dataset reflects reported incidents of crime (with the exception of murders where data exists for each victim) that occurred in the City of Chicago from 2001 to end of October 2018.

Data is extracted from the Chicago Police Department's CLEAR (Citizen Law Enforcement Analysis and Reporting) system. In order to protect the privacy of crime victims, addresses are shown at the block level only and specific locations are not identified.

The dataset is readily obtainable under [data.gov](https://catalog.data.gov/dataset/crimes-2001-to-present-398a4).

## Initial findings

![alt-text](https://github.com/thsis/INFOSYS/blob/master/analysis/ "LSTM on the whole series")

For starters we fit a simple LSTM network with one hidden layer and 4 neurons. The red line is the original data and the blue and green lines depict the fit on the train and test set respectively.

We see that, on the early parts of the time series, the model fits the data already very good, however when looking at the test set the fit becomes a little worse, this means that our initial proof of concept is overfitting the training data.

In spite of the initial overfitting we conclude that the LSTM's forget-gate is indeed capable of automatically setting the irrelevant points in the series to zero, it just needs a little tuning.

Encouraged by these findings we also expect the **GRU** to perform even better, since it is better suited to deal with the problem of both exploding and vanishing gradients.

## Further steps

1. improve fit even further by an extensive grid search.
2. extend the research question in order to provide predictions per district/ward, which may (or may not) prove to be a helpful tool to the authorities trying to make a decision about the allocation of patrol cars.

## Documentation

### Recurrent
```python
Recurrent(self, data, maxlag, cell, cell_neurons=4, num_cells=1, lossfunc='mean_squared_error', optimizer='adam', cellkwargs={}, fitkwargs={}, epochs=100, batch_size=1, train_size=0.8, verbose=True)
```

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
    * `num_cells`: (`int`) number of cells.
    * `lossfunc`: loss-function to be used.
    * `optimizer`: optimizer to be used.
    * `cellkwargs`: (`dict`) `kwargs` to `keras.Sequential.fit`-method.
    * `fitkwargs`: (`dict`) keyword arguments for `cell.`
    * `epochs`: (`int`) number of training-epochs.
    * `batch_size`: (`int`) number of samples per batch.
    * `train_size`: (`int`) propotion of training samples.
      0 < train_size < 1.
    * `verbose`: (`bool`) level of verbosity during training.

#### train
```python
Recurrent.train(self)
```
Train the model based on parameters passed to `__init__`.
#### forecast
```python
Recurrent.forecast(self, X)
```

Calculate predictions.

* Parameters:
    * `X`: `numpy.ndarray` of shape (`n`, `maxlag`), `n` is arbitrary.

* Returns:
    * `out`: predictions of model of shape (`n`, 1).

#### plot_fit
```python
Recurrent.plot_fit(self, savepath=None, **kwargs)
```

Visualize fit of training and test data.

* Parameters:
    * `savepath`: Filename to save figure, not saved if `None`.
    * `**kwargs`: Keyword arguments to `pyplot.subplots`.

* Returns:
    * `(fig, ax)`: Figure and axis, depends on index of `self.data`.
        + if `pandas.DatetimeIndex`: single plot of whole series.
        + if `pandas.MultiIndex`: multiple subplots for each series.
