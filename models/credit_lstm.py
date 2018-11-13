import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler

from keras.models import Sequential
from keras.layers import Dense, LSTM

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

datapath = os.path.join("data", "full.csv")

data = pd.read_csv(datapath, sep=";", header=0)
data.describe()
data.columns


X = data.drop(["T2", "ID"], axis=1).values
y = data["T2"]

ros = RandomOverSampler(random_state=42)
X_res, y_res = ros.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X_res, y_res)

# Edw gamietai




X_train = np.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = np.reshape(X_test.shape[0], 1, X_test.shape[1])

model = Sequential()
model.add(LSTM(4, input_shape=()))
model.add(Dense(1))
model.compile(loss="cross_entropy", optimizer="adam")
model.fit(X_train, y_train, epochs=100, batch_size=1, verbose=2)
X_train.shape


X_train



data.sort_values(["ID", "JAHR"])
data.ID.value_counts("ID")
data.ID.unique().shape
data.shape
