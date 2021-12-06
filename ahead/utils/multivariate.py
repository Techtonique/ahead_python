import os
import numpy as np
import pandas as pd

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector
from datetime import datetime
from rpy2.robjects.vectors import StrVector

base = importr('base')
stats = importr('stats')

def compute_y_mts(df, df_frequency):

    input_series = df.to_numpy()

    """ https://otexts.com/fpp2/ts-objects.html#frequency-of-a-time-series
    Data 	frequency
    Annual 	        1
    Quarterly 	    4
    Monthly 	    12
    Weekly 	        52
    """        
    frequency_choices = {"A": 1,
                         "Y": 1,
                         "BA": 1,
                         "BY": 1,
                         "AS": 1,
                         "AS-JAN": 1,
                         "YS": 1,
                         "BAS": 1,
                         "BYS": 1,
                         "Q": 4,
                         "BQ": 4,
                         "QS": 4,
                         "BQS": 4,
                         "M": 12,
                         "BM": 12, 
                         "CBM": 12,
                         "MS": 12, 
                         "BMS": 12, 
                         "CBMS": 12, 
                         "W": 52}
    
    input_series_tolist = input_series.tolist()
    xx = [item for sublist in input_series_tolist for item in sublist]  
    
    return stats.ts(base.matrix(FloatVector(xx), byrow=True, 
                            nrow=len(input_series_tolist)), 
                            frequency = frequency_choices[df_frequency])
            

def format_multivariate_forecast(n_series, date_formatting, output_dates, horizon, fcast): 

    if (date_formatting == "original"): 
        output_dates_ = [datetime.strftime(output_dates[i], "%Y-%m-%d") for i in range(horizon)]  

    if (date_formatting == "ms"):  
        output_dates_ = [int(datetime.strptime(str(output_dates[i]), "%Y-%m-%d").timestamp()*1000) for i in range(horizon)]

    averages = [] 
    ranges = []

    for j in range(n_series):
        averages_series_j  = []
        ranges_series_j  = []
        for i in range(horizon): 
            date_i = output_dates_[i]
            index_i_j = i+j*horizon     
            averages_series_j.append([date_i, 
                fcast.rx2['mean'][index_i_j]])
            ranges_series_j.append([date_i, 
                fcast.rx2['lower'][index_i_j], fcast.rx2['upper'][index_i_j]])
        averages.append(averages_series_j)
        ranges.append(ranges_series_j) 

    return averages, ranges, output_dates_