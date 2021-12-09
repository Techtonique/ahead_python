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

class Ridge2Regressor():

    def __init__(self, h=5, level=95, 
                 lags=1, nb_hidden=5,
                 nodes_sim="sobol", activation="relu",
                 a=0.01, lambda_1=0.1, lambda_2=0.1, 
                 type_pi = "gaussian", B = 100, 
                 date_formatting="original", seed=123):
        
        self.h = h
        self.level = level
        self.lags = lags
        self.nb_hidden = nb_hidden
        self.nodes_sim = nodes_sim
        self.activation = activation
        self.a = a
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.type_pi = type_pi
        self.B = B
        self.date_formatting=date_formatting
        self.seed = seed
        
        self.averages = None
        self.ranges = None  
        self.output_dates = [] 
        self.result_dfs = None
        self.sims = None 

    def forecast(self, df): 

        self.input_df = df
        n_series = len(df.columns)
        averages = []
        ranges = []

        # obtain dates 'forecast' -----

        output_dates, frequency = umv.compute_output_dates(self.input_df, self.h)                                

        # obtain time series forecast -----
        
        y = mv.compute_y_mts(self.input_df, frequency)
        self.fcast = ahead.ridge2f(y, h=self.h, level=self.level, 
                               lags=self.lags, nb_hidden=self.nb_hidden, 
                               nodes_sim=self.nodes_sim, activ=self.activation, 
                               a = self.a, lambda_1=self.lambda_1, 
                               lambda_2 = self.lambda_2, 
                               type_pi = self.type_pi, B = self.B, 
                               seed=self.seed)

        # result -----
        
        self.averages, self.ranges, self.output_dates = mv.format_multivariate_forecast(n_series=n_series, date_formatting=self.date_formatting, 
        output_dates=output_dates, horizon=self.h, fcast=self.fcast)

        self.result_dfs = tuple(umv.compute_result_df(self.averages[i], self.ranges[i]) for i in range(n_series))

        if self.type_pi == "bootstrap": 
            self.sims = tuple(np.asarray(self.fcast.rx2['sims'][i]) for i in range(self.B))

        return self