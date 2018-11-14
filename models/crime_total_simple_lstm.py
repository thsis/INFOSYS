import os
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import LSTM, Dense

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from matplotlib import pyplot as plt

datapath = os.path.join("data", "crime_total.csv")
dataset = pd.read_csv(datapath, index_col="date",
                      dtype={"crimes_total": np.float64})

dataset["x1"] = dataset.crimes_total.shift(-1)

scaler = MinMaxScaler()
data = scaler.fit_transform(dataset[:-1])
train_size = int(0.8 * len(data))

train_data, test_data = data[:train_size], data[train_size:]

train_data.shape
test_data.shape

X_train, y_train = train_data[:, 1], train_data[:, 0]
X_test, y_test = test_data[:, 1], test_data[:, 0]

# reshape train data to fit into LSTM-layer
X_train = X_train.reshape((X_train.shape[0], 1, 1))
X_test = X_test.reshape((X_test.shape[0], 1, 1))

# Train model
model = Sequential()
model.add(LSTM(4, input_shape=(1, 1)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer="adam")
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)

# predict
train_predictions = model.predict(X_train)
test_predictions = model.predict(X_test)

# transform back
train_data = np.concatenate((y_train.reshape((-1, 1)), train_predictions),
                            axis=1)
train_data = scaler.inverse_transform(train_data)

test_data = np.concatenate((y_test.reshape((-1, 1)), test_predictions), axis=1)
test_data = scaler.inverse_transform(test_data)
train_predictions = train_data[:, 1]
test_predictions = test_data[:, 1]

# calculate test score
test_score = np.sqrt(mean_squared_error(y_test, test_predictions))
print("Mean squared error on test-set: {:3.6}".format(test_score))
plot_data = scaler.inverse_transform(data)
plot_data = plot_data[:, 0]

plot_train = np.empty_like(data[:, 0])
plot_train.fill(np.nan)
plot_train[:train_size] = train_predictions.reshape((-1, ))

plot_test = np.empty_like(data[:, 0])
plot_test.fill(np.nan)
plot_test[train_size:] = test_predictions.reshape((-1, ))

idx = np.arange(0, 6001, 1000)
labels = dataset.index[idx]

plt.figure(figsize=(11, 11))

plt.plot(plot_data, label="data", c="r")
plt.plot(plot_train, label="train", alpha=0.5)
plt.plot(plot_test, label="test", alpha=0.5, c="g")
plt.xticks(idx, labels, rotation=20)
plt.title("Crime-Rate fitted by LSTM")
plt.ylabel("Incidents")
plt.legend()
plt.savefig(os.path.join("models", "total_crime_rate.png"))
