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

from ..utils import univariate as uv
from ..utils import unimultivariate as umv

required_packages = ["ahead"]  # list of required R packages

if all(rpackages.isinstalled(x) for x in required_packages):
    check_packages = True  # True if packages are already installed
else:
    check_packages = False  # False if packages are not installed

if check_packages == False:  # Not installed? Then install.

    packages_to_install = [
        x for x in required_packages if not rpackages.isinstalled(x)
    ]

    if len(packages_to_install) > 0:
        base = importr("base")
        utils = importr("utils")
        base.options(
            repos=base.c(
                techtonique="https://techtonique.r-universe.dev",
                CRAN="https://cloud.r-project.org",
            )
        )
        utils.install_packages(StrVector(packages_to_install))
        check_packages = True


stats = importr("stats")
ahead = importr("ahead")


class EAT(object):
    """Combinations of ETS (exponential smoothing), auto.arima and Theta models

    Parameters:

        h: an integer;
            forecasting horizon

        level: an integer;
            Confidence level for prediction intervals

        weights: a list;
            coefficients assigned to each method in the ensemble

        type_pi: a string;
            Type of prediction interval (currently "gaussian",
            ETS: "E", Arima: "A" or Theta: "T")

        date_formatting: a string;
            Currently:
            - "original": yyyy-mm-dd
            - "ms": milliseconds

    Attributes:

        fcast_: an object;
            raw result from fitting R's `ahead::eatf` through `rpy2`

        averages_: a list;
            mean forecast in a list

        ranges_: a list;
            lower and upper prediction intervals in a list

        output_dates_: a list;
            a list of output dates (associated to forecast)
        
        mean_: a numpy array
            contains series mean forecast as a numpy array 

        lower_: a numpy array 
            contains series lower bound forecast as a numpy array   

        upper_: a numpy array 
            contains series upper bound forecast as a numpy array   

        result_df_: a data frame;
            contains 3 columns, mean forecast, lower + upper
            prediction intervals, and a date index

    Examples:

        ```
            import pandas as pd
            from ahead import EAT

            # Data frame containing the time series
            dataset = {
            'date' : ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01'],
            'value' : [34, 30, 35.6, 33.3, 38.1]}

            df = pd.DataFrame(dataset).set_index('date')
            print(df)

            # univariate time series forecasting
            e1 = EAT(h = 5) # default, equal weights for each model=[1/3, 1/3, 1/3]
            e1.forecast(df)
            print(e1.result_df_)
        ```

    """

    def __init__(
        self,
        h=5,
        level=95,
        weights=[1 / 3, 1 / 3, 1 / 3],
        type_pi="E",
        date_formatting="original",
    ):

        assert len(weights) == 3, "must have 'len(weights) == 3'"

        self.h = h
        self.level = level
        self.weights = weights
        self.type_pi = type_pi
        self.date_formatting = date_formatting

        self.fcast_ = None
        self.averages_ = None
        self.ranges_ = None
        self.output_dates_ = []
        self.mean_ = []
        self.lower_= []
        self.upper_= []
        self.result_df_ = None

    def forecast(self, df):
        """Forecasting method from `EAT` class

        Parameters:

            df: a data frame;
                a data frame containing the input time series (see example)

        """

        self.input_df = df

        # obtain dates 'forecast' -----

        output_dates, frequency = umv.compute_output_dates(
            self.input_df, self.h
        )

        # obtain time series forecast -----

        y = uv.compute_y_ts(df=self.input_df, df_frequency=frequency)

        self.fcast_ = ahead.eatf(
            y=y,
            h=self.h,
            level=self.level,
            type_pi=self.type_pi,
            weights=FloatVector(self.weights),
        )

        # result -----

        (
            self.averages_,
            self.ranges_,
            self.output_dates_,
        ) = uv.format_univariate_forecast(
            date_formatting=self.date_formatting,
            output_dates=output_dates,
            horizon=self.h,
            fcast=self.fcast_,
        )

        self.mean_ = np.asarray(self.fcast_.rx2['mean'])
        self.lower_= np.asarray(self.fcast_.rx2['lower'])
        self.upper_= np.asarray(self.fcast_.rx2['upper'])

        self.result_df_ = umv.compute_result_df(self.averages_, self.ranges_)

        return self
