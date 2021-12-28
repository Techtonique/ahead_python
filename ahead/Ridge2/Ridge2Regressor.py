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


class Ridge2Regressor(object):
    """Random Vector functional link network model with 2 regularization parameters

    Parameters:

        h: an integer;
            forecasting horizon

        level: an integer;
            Confidence level for prediction intervals

        lags: an integer;
            Number of lags

        nb_hidden: an integer;
            Number of nodes in hidden layer

        nodes_sim: an integer;
            Type of simulation for nodes in the hidden layer
            ("sobol", "halton", "unif")

        activation: a string;
            Activation function ("relu", "sigmoid", "tanh",
            "leakyrelu", "elu", "linear")

        a: a float;
            hyperparameter for activation function "leakyrelu", "elu"

        lambda_1: a float;
            Regularization parameter for original predictors

        lambda_2: a float;
            Regularization parameter for transformed predictors

        type_pi: a string;
            Type of prediction interval (currently "gaussian",
            or "bootstrap")

        B: an integer
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

        ```
            import pandas as pd
            from ahead import Ridge2Regressor

            # Data frame containing the time series
            dataset = {
            'date' : ['2001-01-01', '2002-01-01', '2003-01-01', '2004-01-01', '2005-01-01'],
            'series1' : [34, 30, 35.6, 33.3, 38.1],
            'series2' : [4, 5.5, 5.6, 6.3, 5.1],
            'series3' : [100, 100.5, 100.6, 100.2, 100.1]}
            df = pd.DataFrame(dataset).set_index('date')
            print(df)

            # multivariate time series forecasting
            r1 = Ridge2Regressor(h = 5)
            r1.forecast(df)
            print(r1.result_dfs_)
        ```

    """

    def __init__(
        self,
        h=5,
        level=95,
        lags=1,
        nb_hidden=5,
        nodes_sim="sobol",
        activation="relu",
        a=0.01,
        lambda_1=0.1,
        lambda_2=0.1,
        type_pi="gaussian",
        B=100,
        date_formatting="original",
        seed=123,
    ):

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
        """Forecasting method from `Ridge2Regressor` class

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
        self.fcast_ = ahead.ridge2f(
            y,
            h=self.h,
            level=self.level,
            lags=self.lags,
            nb_hidden=self.nb_hidden,
            nodes_sim=self.nodes_sim,
            activ=self.activation,
            a=self.a,
            lambda_1=self.lambda_1,
            lambda_2=self.lambda_2,
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
