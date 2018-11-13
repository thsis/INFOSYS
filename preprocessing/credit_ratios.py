"""
Define Finantial Ratios.abs

Following steps are undertaken:
    - Read cleaned data.
    - Define finantial ratios.
    - Censor values for ratios if they fall outside of
      [0.05-percentile, 0.95-percentile] by the respective value.
"""

import pandas as pd
import numpy as np
import os

data_path = os.path.join("data", "credit_clean.csv")
data = pd.read_csv(data_path, sep=';')

# Define ratios.
data['x1'] = data['VAR22'] / data['VAR6']
data['x2'] = data['VAR22'] / data['VAR16']
data['x3'] = data['VAR21'] / data['VAR6']
data['x4'] = data['VAR21'] / data['VAR16']
data['x5'] = data['VAR20'] / data['VAR6']
data['x6'] = (data['VAR20'] + data['VAR18']) / data['VAR6']
data['x7'] = data['VAR20'] / data['VAR16']
data['x8'] = data['VAR9'] / data['VAR6']
data['x9'] = (data['VAR9'] - data['VAR5']) / \
    (data['VAR6']-data['VAR5']-data['VAR1']-data['VAR8'])
data['x10'] = data['VAR12'] / data['VAR6']
data['x11'] = (data['VAR12'] - data['VAR1']) / data['VAR6']
data['x12'] = (data['VAR12'] + data['VAR13']) / data['VAR6']
data['x13'] = data['VAR14'] / data['VAR6']
data['x14'] = data['VAR20'] / data['VAR19']
data['x15'] = data['VAR1'] / data['VAR6']
data['x16'] = data['VAR1'] / data['VAR12']
data['x17'] = (data['VAR3'] - data['VAR2']) / data['VAR12']
data['x18'] = data['VAR3'] / data['VAR12']
data['x19'] = (data['VAR3'] - data['VAR12']) / data['VAR6']
data['x20'] = data['VAR12'] / (data['VAR12'] + data['VAR13'])
data['x21'] = data['VAR6'] / data['VAR16']
data['x22'] = data['VAR2'] / data['VAR16']
data['x23'] = data['VAR7'] / data['VAR16']
data['x24'] = data['VAR15'] / data['VAR16']
data['x25'] = np.log(data['VAR6'])
data['x26'] = data['VAR23'] / data['VAR2']
data['x27'] = data['VAR24'] / (data['VAR12'] + data['VAR13'])
data['x28'] = data['VAR25'] / data['VAR1']


# Censor outliers: replace values lower than 0.05-quantile by 0.05-quantile,
# and values higher than 0.95-quantile by 0.95-quantile.
ratios = data[['x' + str(i) for i in range(1, 29)]]

perc05 = ratios.quantile(0.05)
lower = (ratios < perc05)
ratios = ratios.mask(lower, perc05, axis=1)

perc95 = ratios.quantile(0.95)
higher = (ratios > perc95)
ratios = ratios.mask(higher, perc95, axis=1)

# Stitch dataframes back together.
data[['x' + str(i) for i in range(1, 29)]] = ratios

# Write data to disk
ratios["T2"] = data["T2"]
ratios_out = os.path.join('data', 'ratios.csv')
ratios.to_csv(ratios_out, sep=';', index=False)

full_out = os.path.join('data', 'full.csv')
data.to_csv(full_out, sep=';', index=False)
