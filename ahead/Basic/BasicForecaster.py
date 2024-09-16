import numpy as np

from ..Base import Base
from ..utils import multivariate as mv
from ..utils import unimultivariate as umv
from .. import config


class BasicForecaster(Base):
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
            "bootstrap" (independent), "blockbootstrap" (circular),
            "movingblockbootstrap")

        block_length: an integer
            length of block for multivariate block bootstrap (`type_pi == blockbootstrap`
            or `type_pi == movingblockbootstrap`)

        B: an integer;
            Number of replications

        date_formatting: a string;
            Currently:
            - "original": yyyy-mm-dd
            - "ms": milliseconds

        seed: an integer;
            reproducibility seed

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
        block_length=5,
        B=100,
        date_formatting="original",
        seed=123,
    ):

        super().__init__(
            h=h,
            level=level,
            seed=seed,
        )

        self.method = method
        self.type_pi = type_pi
        self.block_length = block_length
        self.B = B
        self.date_formatting = date_formatting
        self.input_df = None

        self.fcast_ = None
        self.averages_ = None
        self.ranges_ = None
        self.output_dates_ = []
        self.mean_ = None
        self.lower_ = None
        self.upper_ = None
        self.result_dfs_ = None
        self.sims_ = None

    def forecast(self, df):
        """Forecasting method from `BasicForecaster` class

        Parameters:

            df: a data frame;
                a data frame containing the input time series (see example)

        """

        # get input dates, output dates, number of series, series names, etc.
        self.init_forecasting_params(df)

        # obtain time series object -----
        self.format_input()

        if self.type_pi in ("blockbootstrap", "movingblockbootstrap"):
            assert (
                self.block_length is not None
            ), "For `type_pi in ('blockbootstrap', 'movingblockbootstrap')`, `block_length` must be not None"

        self.get_forecast()

        # result -----
        (
            self.averages_,
            self.ranges_,
            _,
        ) = mv.format_multivariate_forecast(
            n_series=self.n_series,
            date_formatting=self.date_formatting,
            output_dates=self.output_dates_,
            horizon=self.h,
            fcast=self.fcast_,
        )

        self.mean_ = np.asarray(self.fcast_.rx2["mean"])
        self.lower_ = np.asarray(self.fcast_.rx2["lower"])
        self.upper_ = np.asarray(self.fcast_.rx2["upper"])

        self.result_dfs_ = tuple(
            umv.compute_result_df(self.averages_[i], self.ranges_[i])
            for i in range(self.n_series)
        )

        if self.type_pi in (
            "bootstrap",
            "blockbootstrap",
            "movingblockbootstrap",
        ):
            self.sims_ = tuple(
                np.asarray(self.fcast_.rx2["sims"][i]) for i in range(self.B)
            )

        return self
