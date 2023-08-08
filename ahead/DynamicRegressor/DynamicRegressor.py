from subprocess import Popen, PIPE # from https://stackoverflow.com/questions/25329955/check-if-r-is-installed-from-python
import numpy as np

from ..utils import univariate as uv 
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


class DynamicRegressor(object):
    """Dynamic Regression Model adapted from R's `forecast::nnetar`

    Parameters:

        h: an integer;
            forecasting horizon

        level: an integer;
            Confidence level for prediction intervals

        type_pi: a string;
            Type of prediction interval (currently "gaussian",
            ETS: "E", Arima: "A" or Theta: "T")

        date_formatting: a string;
            Currently:
            - "original": yyyy-mm-dd
            - "ms": milliseconds

    Attributes:

        fcast_: an object;
            raw result from fitting R's `ahead::dynrmf` through `rpy2`

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

    ```python
    import pandas as pd
    from ahead import DynamicRegressor

    # Data frame containing the time series
    dataset = {
    'date' : ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01'],
    'value' : [34, 30, 35.6, 33.3, 38.1]}

    df = pd.DataFrame(dataset).set_index('date')
    print(df)

    # univariate time series forecasting
    d1 = DynamicRegressor(h = 5)
    d1.forecast(df)
    print(d1.result_df_)
    ```

    """

    def __init__(self, h=5, level=95, type_pi="E", date_formatting="original"):

        if not R_IS_INSTALLED:
            raise ImportError("R is not installed! \n" + USAGE_MESSAGE)
        
        if not RPY2_IS_INSTALLED:
            raise ImportError(rpy2_error_message + USAGE_MESSAGE)

        self.h = h
        self.level = level
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
        """Forecasting method from `DynamicRegressor` class

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

        self.fcast_ = ahead.dynrmf(
            y=y, h=self.h, level=self.level, type_pi=self.type_pi
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
