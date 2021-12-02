import numpy as np
import pandas as pd
from ahead import DynamicRegressor, EAT
from time import time


# Forecasting horizon
h = 5


# Data frame containing the time series 
dataset = {
    'date' : ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01'],
    'value' : [34, 30, 35.6, 33.3, 38.1],    
}
df = pd.DataFrame(dataset).set_index('date')
print(df)


# univariate ts forecasting 
print("Example 1 -----")
d1 = DynamicRegressor(h = h, date_formatting = "ms")

start = time()
d1.forecast(df)
print(f"Elapsed: {time()-start} \n")
print("averages: \n")
print(d1.averages)
print("\n")
print("ranges: \n")
print(d1.ranges)
print("\n")

print("Example 2 -----")
d2 = DynamicRegressor(h = h, type_pi="T", date_formatting = "original")
start = time()
d2.forecast(df)
print(f"Elapsed: {time()-start} \n")
print("averages: \n")
print(d2.averages)
print("\n")
print("ranges: \n")
print(d2.ranges)
print("\n")

""" print("Example 3 -----")

df2 = pd.read_csv('/Users/t/Documents/sandbox/techtonique-stuff/v0.3.0/nile.csv').set_index('date')
print(df2.head())
print(df2.tail())

#d3 = DynamicRegressor(type_pi="gaussian")
d3 = DynamicRegressor(type_pi="gaussian")
start = time()
d3.forecast(df2, freq=1)
print(f"Elapsed: {time()-start} \n")
print("averages: \n")
print(d3.averages)
print("\n")
print("ranges: \n")
print(d3.ranges)
print("\n") """