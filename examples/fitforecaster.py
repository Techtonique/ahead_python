import os 
import numpy as np
import pandas as pd
from ahead import FitForecaster
from time import time

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

url = "https://raw.githubusercontent.com/Techtonique/"
url += "datasets/main/time_series/univariate/"
url += "a10.csv"

df = pd.read_csv(url)
df.index = pd.DatetimeIndex(df.date) # must have
df.drop(columns=['date'], inplace=True)

# univariate ts forecasting 
print("Example 1 -----")
d1 = FitForecaster()

print(f"before: {d1}")

start = time()
print(d1.fit_forecast(df))
print(f"Elapsed: {time()-start} \n")

print(f"after: {d1.result_dfs_}")