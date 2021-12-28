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

base = importr("base")
stats = importr("stats")
ahead = importr("ahead")


class VAR(object):
    """Vector AutoRegressive model

    Parameters:

        h: an integer;
            forecasting horizon

        level: an integer;
            Confidence level for prediction intervals

        lags: an integer;
            the lag order

        type_VAR: a string;
            Type of deterministic regressors to include
            ("const", "trend", "both", "none")

        date_formatting: a string;
            Currently:
            - "original": yyyy-mm-dd
            - "ms": milliseconds

    Attributes:

        fcast_: an object;
            raw result from fitting R's `ahead::varf` through `rpy2`

        averages_: a list of lists;
            mean forecast in a list for each series

        ranges_: a list of lists;
            lower and upper prediction intervals in a list for each series

        output_dates_: a list;
            a list of output dates (associated to forecast)

        mean_: a numpy array
            contains series mean forecast as a numpy array

        lower_: a numpy array 
            contains series lower bound forecast as a numpy array   

        upper_: a numpy array 
            contains series upper bound forecast as a numpy array   

        result_dfs_: a tuple of data frames;
            each element of the tuple contains 3 columns,
            mean forecast, lower + upper prediction intervals,
            and a date index

    Examples:

        ```
            import pandas as pd
            from ahead import VAR

            # Data frame containing the time series
            dataset = {
            'date' : ['2001-01-01', '2002-01-01', '2003-01-01', '2004-01-01', '2005-01-01'],
            'series1' : [34, 30, 35.6, 33.3, 38.1],
            'series2' : [4, 5.5, 5.6, 6.3, 5.1],
            'series3' : [100, 100.5, 100.6, 100.2, 100.1]}
            df = pd.DataFrame(dataset).set_index('date')
            print(df)

            # multivariate time series forecasting
            v1 = VAR(h = 5, date_formatting = "original", type_VAR="none")
            v1.forecast(df)
            print(v1.result_dfs_)
        ```

    """

    def __init__(
        self, h=5, level=95, lags=1, type_VAR="none", date_formatting="original"
    ):  # type_VAR = "const", "trend", "both", "none"

        assert type_VAR in (
            "const",
            "trend",
            "both",
            "none",
        ), "must have: type_VAR in ('const', 'trend', 'both', 'none')"

        self.h = h
        self.level = level
        self.lags = lags
        self.type_VAR = type_VAR
        self.date_formatting = date_formatting

        self.averages_ = None
        self.ranges_ = None
        self.output_dates_ = []
        self.mean_ = None
        self.lower_ = None
        self.upper_ = None
        self.result_df_s_ = None

    def forecast(self, df):
        """Forecasting method from `VAR` class

        Parameters:

            df: a data frame;
                a data frame containing the input time series (see example)

        """

        self.input_df = df
        n_series = len(df.columns)

        # obtain dates 'forecast' -----

        output_dates, frequency = umv.compute_output_dates(
            self.input_df, self.h
        )

        # obtain time series forecast -----

        y = mv.compute_y_mts(self.input_df, frequency)
        self.fcast_ = ahead.varf(
            y,
            h=self.h,
            level=self.level,
            lags=self.lags,
            type_VAR=self.type_VAR,
        )

        # result -----

        (
            self.averages_,
            self.ranges_,
            self.output_dates_,
        ) = mv.format_multivariate_forecast(
            n_series=n_series,
            date_formatting=self.date_formatting,
            output_dates=output_dates,
            horizon=self.h,
            fcast=self.fcast_,
        )

        self.mean_ = np.asarray(self.fcast_.rx2['mean'])
        self.lower_= np.asarray(self.fcast_.rx2['lower'])
        self.upper_= np.asarray(self.fcast_.rx2['upper'])

        self.result_dfs_ = tuple(
            umv.compute_result_df(self.averages_[i], self.ranges_[i])
            for i in range(n_series)
        )

        return self
