import os
import numpy as np
import pandas as pd

datapath = os.path.join("..", "data", "crimes.zip")
columns = ["ID", "Date", "Block", "Primary Type", "Latitude", "Longitude", "District", "Ward", "Community Area"]
fulldata = pd.read_csv(datapath, usecols=columns)
fulldata.info()

fulldata["Date"] = pd.to_datetime(fulldata["Date"],
                                  format='%m/%d/%Y',
                                  exact=False)
fulldata = fulldata.set_index("Date")

crimes_total = fulldata.groupby(fulldata.index.date).ID.count().reset_index()
crimes_total.to_csv(os.path.join("..", "data", "crime_total.csv"),
                    header=["date", "crimes_total"], index=False)
crimes_total.head(10)

fulldata.District.unique()

crimes_district = fulldata.dropna(subset=["District"])
crimes_district = crimes_district.groupby(["District", crimes_district.index]).ID.count()

new_index = pd.MultiIndex.from_product(crimes_district.index.levels)
crimes_district = crimes_district.reindex(new_index).fillna(0)

assert len(crimes_district) == fulldata.District.nunique() * fulldata.index.nunique()


crimes_district = crimes_district.reset_index()
crimes_district.to_csv(os.path.join("..", "data", "crimes_district.csv"),
                                     header=["District", "Date", "Incidents"],
                       index=False)

