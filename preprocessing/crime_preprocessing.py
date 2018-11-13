import os
import pandas as pd

"""
Data: https://catalog.data.gov/dataset/crimes-2001-to-present-398a4
"""

datapath = os.path.join("data", "crimes.csv")
data = pd.read_csv(datapath, header=0, parse_dates=["Date"], index_col="Date")

# Count all reported crime as the same.
X = data.groupby(data.index.date).ID.count()

# Write to csv.
X.reset_index().to_csv(os.path.join("data", "crime_total.csv"),
                       header=["date", "crimes_total"], index=False)
