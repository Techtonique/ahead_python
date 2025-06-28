import os 
import numpy as np
import pandas as pd
from ahead import MLARCH
from time import time

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# Forecasting horizon
h = 5


# Data frame containing the time series 
df = pd.read_csv("https://raw.githubusercontent.com/Techtonique/datasets/refs/heads/main/time_series/univariate/AirPassengers.csv").set_index('date')
df.index = pd.DatetimeIndex(df.index)
print(df)



print("Example 2 -----")
d2 = MLARCH(h = h, date_formatting = "original")
start = time()
d2.forecast(df)
print(f"Elapsed: {time()-start} \n")
print("averages: \n")
print(d2.averages_)
print("\n")
print("ranges: \n")
print(d2.ranges_)
print("\n")

