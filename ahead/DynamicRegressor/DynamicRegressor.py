import os
import numpy as np
import pandas as pd

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector
from datetime import datetime
from rpy2.robjects.vectors import StrVector

from ..utils import univariate as uv
from ..utils import unimultivariate as umv

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

class DynamicRegressor():

    def __init__(self, h=5, level=95,                  
                 type_pi="E", date_formatting="original"):
        
        self.h = h
        self.level = level
        self.type_pi = type_pi
        self.date_formatting=date_formatting

        self.fcast = None
        self.averages = None
        self.ranges = None   
        self.output_dates = []      
        self.result_df = None

    def forecast(self, df):            
        
        self.input_df = df

        # obtain dates 'forecast' -----        

        output_dates, frequency = umv.compute_output_dates(self.input_df, self.h)                                

        # obtain time series forecast -----

        y = uv.compute_y_ts(df = self.input_df, df_frequency=frequency)

        self.fcast = ahead.dynrmf(y=y, h=self.h, level=self.level, type_pi=self.type_pi)         

        # result -----
        
        self.averages, self.ranges, self.output_dates = uv.format_univariate_forecast(date_formatting=self.date_formatting, 
        output_dates=output_dates, horizon=self.h, fcast=self.fcast)

        self.result_df = umv.compute_result_df(self.averages, self.ranges)

        return self         
                            
