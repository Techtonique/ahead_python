import numpy as np
from .. import config

from ..utils import univariate as uv
from ..utils import unimultivariate as umv


class ArmaGarch(object):
    """ARMA(1, 1)-GARCH(1, 1) forecasting (with simulation)

    Parameters:

        h: an integer;
            forecasting horizon

        level: an integer;
            Confidence level for prediction intervals

        B: an integer;
            number of simulations for R's `stats::arima.sim`

        cl: an integer;
            the number of clusters for parallel execution (done in R /!\)

        dist: a string;
            distribution of innovations ("student" or "gaussian")

        seed: an integer;
            reproducibility seed

        date_formatting: a string;
            Currently:
            - "original": yyyy-mm-dd
            - "ms": milliseconds

    Attributes:

        fcast_: an object;
            raw result from fitting R's `ahead::armagarchf` through `rpy2`

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

        sims_: a numpy array
            forecasting simulations

    """

    def __init__(
        self,
        h=5,
        level=95,
        B=250,
        cl=1,
        dist="student",
        seed=123,
        date_formatting="original",
    ):

        self.h = h
        self.level = level
        self.B = B
        self.cl = cl
        self.dist = dist
        self.seed = seed
        self.date_formatting = date_formatting
        self.input_df = None

        self.fcast_ = None
        self.averages_ = None
        self.ranges_ = None
        self.output_dates_ = []
        self.mean_ = []
        self.lower_ = []
        self.upper_ = []
        self.result_df_ = None
        self.sims_ = None

    def forecast(self, df):
        """Forecasting method from `ArmaGarch` class

        Parameters:

            df: a data frame;
                a data frame containing the input time series (see example)

        """

        # get input dates, output dates, number of series, series names, etc.
        self.init_forecasting_params(df)

        # obtain time series object -----
        self.format_input()

        self.get_forecast("armagarch")

        # result -----
        (
            self.averages_,
            self.ranges_,
            self.output_dates_,
        ) = uv.format_univariate_forecast(
            date_formatting=self.date_formatting,
            output_dates=self.output_dates_,
            horizon=self.h,
            fcast=self.fcast_,
        )

        self.mean_ = np.asarray(self.fcast_.rx2["mean"])
        self.lower_ = np.asarray(self.fcast_.rx2["lower"])
        self.upper_ = np.asarray(self.fcast_.rx2["upper"])

        self.result_df_ = umv.compute_result_df(self.averages_, self.ranges_)

        self.sims_ = np.asarray(self.fcast_.rx2["sims"])

        return self
