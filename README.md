# Seminar Information Systems

This repository contains code for illustrating the *Long Short Term Memory* (**LSTM**) and *Gated Recurrent Unit* (**GRU**) architectures for neural networks as well as comparing them to traditional *Recurrent Neural Networks* (**RNN**).

## Crime Data

This dataset reflects reported incidents of crime (with the exception of murders where data exists for each victim) that occurred in the City of Chicago from 2001 to end of October 2018.

Data is extracted from the Chicago Police Department's CLEAR (Citizen Law Enforcement Analysis and Reporting) system. In order to protect the privacy of crime victims, addresses are shown at the block level only and specific locations are not identified.

The dataset is readily obtainable under [data.gov](https://catalog.data.gov/dataset/crimes-2001-to-present-398a4).

## Initial findings

![alt-text](https://github.com/thsis/INFOSYS/blob/master/models/total_crime_rate.png "First fit using LSTM and only one lagged variable")

For starters we fit a simple LSTM network with one hidden layer and 4 neurons. The red line is the original data and the blue and green lines depict the fit on the train and test set respectively.

We see that, on the early parts of the time series, the model fits the data only poorly - but after a certain time period the fit becomes better, which even generalizes to the test set.

We conclude that the LSTM's forget-gate automatically set the early points in the series to zero and used the more recent observation to generate predictions.

## Further steps

1. improve fit even further by an extensive grid search.
2. extend the research question in order to provide predictions per district/ward, which may (or may not) prove to be a helpful tool to the authorities trying to make a decision about the allocation of patrol cars.
