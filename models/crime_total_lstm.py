import os
import numpy as np
import pandas as pd

from models.lstm import LSTMNET
from sklearn.metrics import mean_squared_error
from hyperopt import tpe, hp, fmin, STATUS_OK, Trials
from hyperopt.pyll.base import scope

from matplotlib import pyplot as plt

datapath = os.path.join("data", "crime_total.csv")
dataset = pd.read_csv(datapath, index_col="date",
                      dtype={"crimes_total": np.float32})


def objective(params):
    model = LSTMNET(dataset, **params)
    model.train()
    predictions = model.predict(model.X_test)
    loss = mean_squared_error(y_true=model.y_test, y_pred=predictions)
    return {'loss': loss, 'status': STATUS_OK}


def plot_trials(trials, paramspace):
    """Plot path of hyperopt-search for analysis."""
    parameters = list(paramspace.keys())
    cols = len(parameters)
    fig, axes = plt.subplots(nrows=1, ncols=cols, figsize=(20, 5))

    for i, val in enumerate(parameters):
        xs = np.array([t['misc']['vals'][val] for t in trials.trials]).ravel()
        ys = [t['result']['loss'] for t in trials.trials]
        xs, ys = zip(*sorted(zip(xs, ys)))
        axes[i].scatter(xs, ys, s=20, linewidth=0.01, alpha=0.25)
        axes[i].set_title(val)
        axes[i].set_ylim([0, 0.002])

    return fig, axes


if __name__ == "__main__":
    paramspace = {
        "maxlag": scope.int(hp.quniform("maxlag", 0, 365, 1)),
        "lstm_neurons": scope.int(hp.quniform("lstm_neurons", 1, 30, 1)),
        "batch_size": scope.int(hp.quniform("batch_size", 1, 10, 1))
    }

    trials = Trials()
    best = fmin(fn=objective,
                space=paramspace,
                algo=tpe.suggest,
                trials=trials,
                max_evals=1000)

    # Fix type of optimal parameters
    best = {key: int(val) for key, val in best.items()}

    fig, axes = plot_trials(trials=trials, paramspace=paramspace)
    plt.savefig(os.path.join("models", "hyperopt-search.png"))

    model = LSTMNET(data=dataset, **best)
    model.train()
    model.plot_fit()
    plt.savefig(os.path.join("models", "lstm-hyperopt-fit.png"))
