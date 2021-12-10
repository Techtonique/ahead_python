import numpy as np
import pandas as pd
from ahead import DynamicRegressor, EAT
from time import time

import warnings
import itertools
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


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
print(d1.averages_)
print("\n")
print("ranges: \n")
print(d1.ranges_)
print("\n")

print("Example 2 -----")
d2 = DynamicRegressor(h = h, type_pi="T", date_formatting = "original")
start = time()
d2.forecast(df)
print(f"Elapsed: {time()-start} \n")
print("averages: \n")
print(d2.averages_)
print("\n")
print("ranges: \n")
print(d2.ranges_)
print("\n")

print("Example 3 -----")
# https://stackoverflow.com/questions/42098126/mac-osx-python-ssl-sslerror-ssl-certificate-verify-failed-certificate-verify/42098127#42098127
# compared to > ahead::dynrmf(y=window(Nile, start=1919), h=10, level=95, type_pi="gaussian")
url = "https://raw.githubusercontent.com/Techtonique/ahead_python/main/datasets/nile.csv"
df2 = pd.read_csv(url)
df2 = df2.set_index('date')
print(df2.head())
print(df2.tail())


d3 = DynamicRegressor(type_pi="gaussian", h=10)
start = time()
d3.forecast(df2)
print(f"Elapsed: {time()-start} \n")
print("averages: \n")
print(d3.averages_)
print("\n")
print("ranges: \n")
print(d3.ranges_)
print("\n")
print("data frame result: \n")
print("\n")
print(d3.result_df_)
print("\n")
print(d3.result_df_.loc[:,"lower"])
print("\n")
print(d3.input_df)
