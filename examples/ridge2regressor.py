import numpy as np
import pandas as pd
from ahead import Ridge2Regressor
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
d1 = Ridge2Regressor(h = h, date_formatting = "ms")

start = time()
d1.forecast(df)
print(f"Elapsed: {time()-start} \n")
print("averages: \n")
print(d1.averages[0])
print("\n")
print(d1.averages[1])
print("\n")
print(d1.averages[2])
print("\n")
print("ranges: \n")
print(d1.ranges[0])
print("\n")
print(d1.ranges[1])
print("\n")
print(d1.ranges[2])
print("\n")

print("Example 2 -----")
d2 = Ridge2Regressor(h = h, date_formatting = "original")
start = time()
d2.forecast(df)
print(f"Elapsed: {time()-start} \n")
print(d2.averages[0])
print("\n")
print(d2.averages[1])
print("\n")
print(d2.averages[2])
print("\n")
print("ranges: \n")
print(d2.ranges[0])
print("\n")
print(d2.ranges[1])
print("\n")
print(d2.ranges[2])
print("\n")


print("Example 3 -----")

d3 = Ridge2Regressor(h = h, date_formatting = "original", 
type_pi="bootstrap", B=3)

start = time()
d3.forecast(df)
print(f"Elapsed: {time()-start} \n")

print(d3.fcast.rx2['mean'])
print(d3.averages[1])
print(np.asarray(d3.fcast.rx2['mean']))

print(d3.fcast.rx2['sims'][0])
res = np.asarray(d3.fcast.rx2['sims'][1])
print(res)
print(res.shape)
print(res[0, 1])

print("\n")
print(d3.sims)

print("\n")
print(d3.output_dates)
