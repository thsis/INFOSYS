import os
import itertools
import numpy as np
import pandas as pd
from mpl_toolkits.basemap import Basemap
from matplotlib import pyplot as plt
from tqdm import tqdm

datapath = os.path.join("data", "crimes.zip")
columns = ["ID", "Date", "Block", "Primary Type", "Latitude", "Longitude",
           "District", "Ward", "Community Area"]

fulldata = pd.read_csv(datapath, usecols=columns)
fulldata["Date"] = pd.to_datetime(fulldata["Date"],
                                  format='%m/%d/%Y',
                                  exact=False)
fulldata = fulldata.set_index("Date")


primary_type_counts = fulldata.groupby("Primary Type").ID.count()
primary_type_counts = primary_type_counts.sort_values(ascending=False)


def plot_chicago(data, savepath, lowerlon=-88.0, upperlon=-87.4,
                 lowerlat=41.62, upperlat=42.05, size=(15, 10), **kwargs):
    """Overlay chicago map with scatterplot."""
    shapepath = os.path.join("data", "US-shapedata", "cb_2017_us_zcta510_500k")
    fig = plt.gcf()
    fig.set_size_inches(*size)

    chicago = Basemap(llcrnrlon=lowerlon,
                      llcrnrlat=lowerlat,
                      urcrnrlon=upperlon,
                      urcrnrlat=upperlat,
                      projection="lcc",
                      resolution="c",
                      lat_0=lowerlat,
                      lat_1=upperlat,
                      lon_0=lowerlon,
                      lon_1=upperlon)

    chicago.readshapefile(shapepath, "state")
    x, y = chicago(data["Longitude"].values, data["Latitude"].values)
    plt.scatter(x, y, **kwargs)
    plt.savefig(savepath)
    return fig


def get_subset(data, year, crime=None):
    """Extract data that matches a specific type of crime and year."""
    if crime is None:
        out = data.loc[data.index.year == year]
        title = "crime_total_" + str(year) + ".png"
    else:
        locator = (data.index.year == year) & (data["Primary Type"] == crime)
        out = data.loc[locator]
        category = crime.lower().replace(" ", "_")
        title = "crime" + "_" + category + "_" + str(year) + ".png"

    if len(out):
        alpha = max(1/len(out), 0.01)
    else:
        alpha = 1

    return title, out, alpha


# Full dataset
plot_chicago(fulldata,
             os.path.join("preprocessing", "maps", "crime_total.png"),
             alpha=0.01)
plt.clf()

# Assault
assaults = fulldata[fulldata["Primary Type"] == "ASSAULT"]
plot_chicago(assaults,
             os.path.join("preprocessing", "maps", "crime_assaults.png"),
             color="brown", alpha=0.01)
plt.clf()

# Automatically create plots:
categories = [None] + primary_type_counts.index.tolist()
cmap = plt.get_cmap('tab20')
colors = cmap(np.linspace(0, 1, len(categories)))
colormap = {c: col for c, col in zip(categories, colors)}
iterations = list(itertools.product(categories, range(2001, 2018)))

for c, y in tqdm(iterations):
    try:
        title, subset, alpha = get_subset(fulldata, year=y, crime=c)
        _ = plot_chicago(subset,
                         os.path.join("preprocessing", "maps", title),
                         alpha=alpha, color=colormap[c])
    except Exception:
        continue
    finally:
        plt.clf()
