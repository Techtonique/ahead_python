import numpy as np
import pandas as pd
from ahead import BasicForecaster
from time import time

# Forecasting horizon
h = 5

# Data frame containing the time series 
dataset = {
 'date' : ['2001-01-01', '2002-01-01', '2003-01-01', '2004-01-01', '2005-01-01'],
 'series1' : [34, 30, 35.6, 33.3, 38.1],    
 'series2' : [4, 5.5, 5.6, 6.3, 5.1],
 'series3' : [100, 100.5, 100.6, 100.2, 100.1]}
df = pd.DataFrame(dataset).set_index('date')

# univariate ts forecasting 
print("Example 1 -----")
d1 = BasicForecaster(h = h, date_formatting = "ms")

start = time()
d1.forecast(df)
print(f"Elapsed: {time()-start} \n")
print("averages: \n")
print(d1.averages_[0])
print("\n")
print(d1.averages_[1])
print("\n")
print(d1.averages_[2])
print("\n")
print("ranges: \n")
print(d1.ranges_[0])
print("\n")
print(d1.ranges_[1])
print("\n")
print(d1.ranges_[2])
print("\n")

print("Example 2 -----")
d2 = BasicForecaster(h = h, date_formatting = "original")
start = time()
d2.forecast(df)
print(f"Elapsed: {time()-start} \n")
print(d2.averages_[0])
print("\n")
print(d2.averages_[1])
print("\n")
print(d2.averages_[2])
print("\n")
print("ranges: \n")
print(d2.ranges_[0])
print("\n")
print(d2.ranges_[1])
print("\n")
print(d2.ranges_[2])
print("\n")


print("Example 3 -----")

d3 = BasicForecaster(h = h, date_formatting = "original", 
type_pi="bootstrap", B=5)

start = time()
d3.forecast(df)
print(f"Elapsed: {time()-start} \n")

print(d3.fcast_.rx2['mean'])
print(d3.averages_[1])
print(np.asarray(d3.fcast_.rx2['mean']))

print(d3.fcast_.rx2['sims'][0])
res = np.asarray(d3.fcast_.rx2['sims'][1])
print(res)
print(res.shape)
print(res[0, 1])

print("\n result_dfs_: \n")
print(d3.result_dfs_)

print("\n sims_: \n")
print(d3.sims_)

print("\n output_dates_: \n")
print(d3.output_dates_)

print("\n mean, lower, upper as numpy arrays: \n")
print(d3.mean_)
print(d3.lower_)
print(d3.upper_)
