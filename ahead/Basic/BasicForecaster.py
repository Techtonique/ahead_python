from subprocess import Popen, PIPE # from https://stackoverflow.com/questions/25329955/check-if-r-is-installed-from-python
import numpy as np

from ..utils import multivariate as mv
from ..utils import unimultivariate as umv

proc = Popen(["which", "R"], stdout=PIPE, stderr=PIPE)
R_IS_INSTALLED = proc.wait() == 0

try: 
    import rpy2.robjects.packages as rpackages
    from rpy2.robjects.packages import importr    
    from rpy2.robjects.vectors import StrVector
    from rpy2 import rinterface, robjects
    from rpy2.rinterface_lib import callbacks
    from rpy2.rinterface_lib.embedded import RRuntimeError
except ImportError as e: 
    rpy2_error_message = str(e)
    RPY2_IS_INSTALLED = False
else: 
    RPY2_IS_INSTALLED = True

USAGE_MESSAGE = """
This Python class, BasicForecaster, is based on R package 'ahead' (https://techtonique.github.io/ahead/). 
You need to install R (https://www.r-project.org/) and rpy2 (https://pypi.org/project/rpy2/).

Then, install R package 'ahead' (if necessary): 
>> R -e 'options(repos = c(techtonique = 'https://techtonique.r-universe.dev',
    CRAN = 'https://cloud.r-project.org'))'
>> R -e 'install.packages("ahead")'    
"""

required_packages = ["ahead"]  # list of required R packages

if all(rpackages.isinstalled(x) for x in required_packages):
    CHECK_PACKAGES = True  # True if packages are already installed
else:
    CHECK_PACKAGES = False  # False if packages are not installed

if CHECK_PACKAGES == False:  # Not installed? Then install.

    packages_to_install = [
        x for x in required_packages if not rpackages.isinstalled(x)
    ]

    base = importr("base")
    utils = importr("utils")
    if len(packages_to_install) > 0:        
        base.options(
            repos=base.c(
                techtonique="https://techtonique.r-universe.dev",
                CRAN="https://cloud.r-project.org",
            )
        )
        utils.install_packages(StrVector(packages_to_install))                

ahead = importr("ahead")
CHECK_PACKAGES = True


class BasicForecaster(object):
    """Basic forecasting functions for multivariate time series (mean, median, random walk)

    Parameters:

        h: an integer;
            forecasting horizon

        level: an integer;
            Confidence level for prediction intervals

        method: a string;
            Forecasting method, either "mean", "median", or random walk ("rw")    

        type_pi: a string;
            Type of prediction interval (currently "gaussian",
            or "bootstrap")

        B: an integer;
            Number of bootstrap replications for `type_pi == bootstrap`

        date_formatting: a string;
            Currently:
            - "original": yyyy-mm-dd
            - "ms": milliseconds

        seed: an integer;
            reproducibility seed for type_pi == 'bootstrap'

    Attributes:

        fcast_: an object;
            raw result from fitting R's `ahead::ridge2f` through `rpy2`

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

        sims_: currently a tuple of numpy arrays
            for `type_pi == bootstrap`, simulations for each series

    Examples:

    ```python
    import pandas as pd
    from ahead import BasicForecaster

    # Data frame containing the time series
    dataset = {
    'date' : ['2001-01-01', '2002-01-01', '2003-01-01', '2004-01-01', '2005-01-01'],
    'series1' : [34, 30, 35.6, 33.3, 38.1],
    'series2' : [4, 5.5, 5.6, 6.3, 5.1],
    'series3' : [100, 100.5, 100.6, 100.2, 100.1]}
    df = pd.DataFrame(dataset).set_index('date')
    print(df)

    # multivariate time series forecasting
    r1 = BasicForecaster(h = 5)
    r1.forecast(df)
    print(r1.result_dfs_)
    ```

    """

    def __init__(
        self,
        h=5,
        level=95,
        method="mean",
        type_pi="gaussian",
        B=100,        
        date_formatting="original",
        seed=123,
    ):

        if not R_IS_INSTALLED:
            raise ImportError("R is not installed! \n" + USAGE_MESSAGE)
        
        if not RPY2_IS_INSTALLED:
            raise ImportError(rpy2_error_message + USAGE_MESSAGE)

        self.h = h
        self.level = level
        self.method = method
        self.type_pi = type_pi
        self.B = B
        self.date_formatting = date_formatting
        self.seed = seed

        self.averages_ = None
        self.ranges_ = None
        self.output_dates_ = []
        self.mean_ = None
        self.lower_ = None
        self.upper_ = None
        self.result_df_s_ = None
        self.sims_ = None

    def forecast(self, df):
        """Forecasting method from `BasicForecaster` class

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
        self.fcast_ = ahead.basicf(
            y,
            h=self.h,
            level=self.level,
            method=self.method,
            type_pi=self.type_pi,
            B=self.B,
            seed=self.seed,
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

        if self.type_pi == "bootstrap":
            self.sims_ = tuple(
                np.asarray(self.fcast_.rx2["sims"][i]) for i in range(self.B)
            )

        return self
