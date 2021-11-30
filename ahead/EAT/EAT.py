import os
import numpy as np
import pandas as pd

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector
from datetime import datetime
from rpy2.robjects.packages import importr
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


stats = importr('stats')
ahead = importr('ahead')

class EAT():

    def __init__(self, h=5, level=95, 
                 weights=[1/3, 1/3, 1/3], 
                 type_pi="E", date_formatting="original"):

        assert len(weights) == 3, "must have 'len(weights) == 3'"
        
        self.h = h
        self.level = level
        self.weights = weights
        self.type_pi = type_pi
        self.date_formatting=date_formatting

        self.fcast = None
        self.averages = None
        self.ranges = None        


    def forecast(self, df, freq=None):            
        
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

        input_series = df.values
        if freq is None: 
            y = stats.ts(FloatVector([item for sublist in input_series.tolist() for item in sublist]))
        else:     
            y = stats.ts(FloatVector([item for sublist in input_series.tolist() for item in sublist]), 
            frequency=freq)
        self.fcast = ahead.eatf(y=y, h=self.h, level=self.level, type_pi=self.type_pi,
                                weights=FloatVector(self.weights))        


        # result -----

        # to be put in utils/ as a function (DRY)

        if (self.date_formatting == "original"): 
            fcast_dates = [datetime.strftime(output_dates[i], "%Y-%m-%d") for i in range(self.h)]            
            self.averages = [[fcast_dates[i], self.fcast.rx2['mean'][i]] for i in range(self.h)]
            self.ranges = [[fcast_dates[i], self.fcast.rx2['lower'][i], self.fcast.rx2['upper'][i]] for i in range(self.h)]            
            return self         
        
        if (self.date_formatting == "ms"):  
            fcast_dates_ms = [int(datetime.strptime(str(output_dates[i]), "%Y-%m-%d").timestamp()*1000) for i in range(self.h)]
            self.averages = [[fcast_dates_ms[i], self.fcast.rx2['mean'][i]] for i in range(self.h)]
            self.ranges = [[fcast_dates_ms[i], self.fcast.rx2['lower'][i], self.fcast.rx2['upper'][i]] for i in range(self.h)]        
            return self         
