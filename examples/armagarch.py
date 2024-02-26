import os 
import numpy as np
import pandas as pd
from ahead import ArmaGarch
from time import time

print(f"\n ----- Running: {os.path.basename(__file__)}... ----- \n")

# Forecasting horizon
h = 5

# univariate ts forecasting 
print("Example 1 -----")
d1 = ArmaGarch(h = h)
print(d1.__module__)
print(dir(d1))

