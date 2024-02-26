import os 
import numpy as np
import pandas as pd
from ahead import EAT
from time import time

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# Forecasting horizon
h = 5


# Data frame containing the time series 
dataset = {
    'date' : ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01'],
    'value' : [34, 30, 35.6, 33.3, 38.1],    
}
df = pd.DataFrame(dataset).set_index('date')
df.index = pd.DatetimeIndex(df.index)
print(df)


# univariate ts forecasting 
print("Example 1 -----")
e1 = EAT(h = h, weights = [0.3, 0.4, 0.3], type_pi="T", date_formatting = "ms")

start = time()
e1.forecast(df)
print(f"Elapsed: {time()-start} \n")
print("averages: \n")
print(e1.averages_)
print("\n")
print("ranges: \n")
print(e1.ranges_)
print("\n")
print(e1.fcast_.rx2['mean'])
print(e1.fcast_.rx2['lower'])
print(e1.fcast_.rx2['upper'])

print("Example 2 -----")
e2 = EAT(h = h, type_pi="T", date_formatting = "original")
start = time()
e2.forecast(df)
print(f"Elapsed: {time()-start} \n")
print("averages: \n")
print(e2.averages_)
print("\n")
print("ranges: \n")
print(e2.ranges_)
print("\n")
print(e2.result_df_)
print("\n")
print(e2.mean_)
print(e2.lower_)
print(e2.upper_)
