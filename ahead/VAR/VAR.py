import os
import numpy as np
import pandas as pd

from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector, numpy2ri
import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from datetime import datetime
from rpy2.robjects.vectors import StrVector

from ..utils import multivariate as mv
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
        self.output_dates = []
        self.result_dfs = None

    def forecast(self, df): 

        self.input_df = df
        n_series = len(df.columns)
        averages = []
        ranges = []

        # obtain dates 'forecast' -----

        output_dates, frequency = umv.compute_output_dates(self.input_df, self.h)                                

        # obtain time series forecast -----
                    
        y = mv.compute_y_mts(self.input_df, frequency)
        self.fcast = ahead.varf(y, h=self.h, level=self.level, 
                               lags=self.lags, type_VAR = self.type_VAR)

        # result -----

        self.averages, self.ranges, self.output_dates = mv.format_multivariate_forecast(n_series=n_series, date_formatting=self.date_formatting, 
        output_dates=output_dates, horizon=self.h, fcast=self.fcast)

        self.result_dfs = tuple(umv.compute_result_df(self.averages[i], self.ranges[i]) for i in range(n_series))

        return self