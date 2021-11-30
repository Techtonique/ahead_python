import os
import numpy as np
import pandas as pd

from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector, numpy2ri
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from datetime import datetime
from rpy2.robjects.vectors import StrVector


required_packages = ['ahead'] # list of required R packages 

if all(rpackages.isinstalled(x) for x in required_packages):
    check_packages = True # True if packages are already installed 
else:
   check_packages = False # False if packages are not installed 

if check_packages == False: # Not installed? Then install.

    packages_to_install = [x for x in required_packages if not rpackages.isinstalled(x)]
    
    if len(packages_to_install) > 0:
        base = importr('base')
        utils = importr('utils')
        base.options(repos = base.c(techtonique = 'https://techtonique.r-universe.dev', 
                                    CRAN = 'https://cloud.r-project.org'))
        utils.install_packages(StrVector(packages_to_install))
        check_packages = True 

base = importr('base')
stats = importr('stats')
ahead = importr('ahead')

class VAR():

    def __init__(self, h=5, level=95, 
                 lags=1, type_VAR="none", 
                 date_formatting="original"): # type_VAR = "const", "trend", "both", "none"


        assert type_VAR in ("const", "trend", "both", "none"),\
         "must have: type_VAR in ('const', 'trend', 'both', 'none')"
        
        self.h = h
        self.level = level
        self.lags = lags
        self.type_VAR = type_VAR
        self.date_formatting=date_formatting
        
        self.averages = None
        self.ranges = None   


    def forecast(self, df): 

        n_series = len(df.columns)
        averages = []
        ranges = []

        # obtain dates 'forecast' -----

        # to be put in utils/ as a function (DRY)

        input_dates = df.index.values 
        n_input_dates = len(input_dates)        
        
        frequency = pd.infer_freq(pd.DatetimeIndex(input_dates))
        output_dates = np.delete(pd.date_range(start=input_dates[-1], 
            periods=self.h+1, freq=frequency).values, 0).tolist()  
        
        df_output_dates = pd.DataFrame({'date': output_dates})
        output_dates = pd.to_datetime(df_output_dates['date']).dt.date                                


        # obtain time series forecast -----

        # to be put in utils/ as a function (DRY)

        input_series = df.values       
        input_series_tolist = input_series.tolist()
        xx = [item for sublist in input_series_tolist for item in sublist]  
        m = stats.ts(base.matrix(FloatVector(xx), byrow=True, 
                                 nrow=len(input_series_tolist)))      

        self.fcast = ahead.varf(m, h=self.h, level=self.level, 
                               lags=self.lags, type_VAR = self.type_VAR)

        # result -----
        
        if self.date_formatting == "original": 
            # to be put in utils/ as a function (DRY)
            for j in range(n_series):
                averages_series_j  = []
                ranges_series_j  = []
                for i in range(self.h): 
                    date_i = datetime.strftime(output_dates[i], "%Y-%m-%d")    
                    index_i_j = i+j*self.h       
                    averages_series_j.append([date_i, 
                        self.fcast.rx2['mean'][index_i_j]])
                    ranges_series_j.append([date_i, 
                        self.fcast.rx2['lower'][index_i_j], self.fcast.rx2['upper'][index_i_j]])
                averages.append(averages_series_j)
                ranges.append(ranges_series_j) 


        if self.date_formatting == "ms": 
            # to be put in utils/ as a function (DRY)
            for j in range(n_series):
                averages_series_j  = []
                ranges_series_j  = []
                for i in range(self.h): 
                    date_i = int(datetime.strptime(str(output_dates[i]), "%Y-%m-%d").timestamp()*1000)
                    index_i_j = i+j*self.h            
                    averages_series_j.append([date_i, 
                        self.fcast.rx2['mean'][index_i_j]])
                    ranges_series_j.append([date_i, 
                        self.fcast.rx2['lower'][index_i_j], self.fcast.rx2['upper'][index_i_j]])
                averages.append(averages_series_j)
                ranges.append(ranges_series_j) 


        self.averages = averages
        self.ranges = ranges                         

        return self