import os 
import numpy as np
import pandas as pd
from ahead import Ridge2Regressor
from time import time


print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# Forecasting horizon
h = 20


# Data frame containing the time series 
df = pd.read_csv("https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/time_series/univariate/AirPassengers.csv").set_index('date')
df.index = pd.DatetimeIndex(df.index)
print(df)

# univariate ts forecasting 
print("Example 1 -----")
d1 = Ridge2Regressor(h = h, date_formatting = "original")

start = time()
d1.forecast(df)
print(f"Elapsed: {time()-start} \n")
print("averages: \n")
print(d1.averages_)
print("\n")
print("ranges: \n")
print(d1.ranges_)
print("\n")
d1.plot()

print("Example 2 -----")

d3 = Ridge2Regressor(h = h, date_formatting = "original", 
type_pi="bootstrap", B=5)

start = time()
d3.forecast(df)
print(f"Elapsed: {time()-start} \n")

print(d3.fcast_.rx2['mean'])
print(d3.averages_[1])
print(np.asarray(d3.fcast_.rx2['mean']))

print(d3.fcast_.rx2['sims'][0])
print(d3.fcast_.rx2['sims'][1])
d3.plot()