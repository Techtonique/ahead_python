import matplotlib.pyplot as plt
import numpy as np
import rpy2.robjects.conversion as cv
from rpy2.robjects import (
    FloatVector,
    ListVector,
    numpy2ri,
    r,
)

from ..Base import Base
from ..utils import multivariate as mv
from ..utils import univariate as uv
from ..utils import unimultivariate as umv
from .. import config


class Ridge2Regressor(Base):
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

        dropout: a float;
            dropout regularization parameter (dropping nodes in hidden layer)

        type_pi: a string;
            Type of prediction interval (currently "gaussian",
            "bootstrap", (circular) "blockbootstrap", "movingblockbootstrap", "rvinecopula", 
            "conformal-split", "conformal-bootstrap", "conformal-block-bootstrap")

        block_length: an integer
            length of block for multivariate block bootstrap (`type_pi == blockbootstrap` or
            `type_pi == movingblockbootstrap`)

        margins: a string;
            distribution of residuals' marginals for `type_pi == rvinecopula`: "empirical" (default),
            "gaussian"

        B: an integer;
            Number of bootstrap replications for `type_pi == bootstrap`, "blockbootstrap",
            "movingblockbootstrap", or "rvinecopula"

        type_aggregation: a string;
            Type of aggregation, ONLY for bootstrapping; either "mean" or "median"

        centers: an integer;
            Number of clusters for \code{type_clustering}

        type_clustering: a string;
            "kmeans" (K-Means clustering) or "hclust" (Hierarchical clustering)

        cl: an integer;
            The number of clusters for parallel execution (done in R), for `type_pi == bootstrap`

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
        lags=15,
        nb_hidden=5,
        nodes_sim="sobol",
        activation="relu",
        a=0.01,
        lambda_1=0.1,
        lambda_2=0.1,
        dropout=0,
        type_pi="gaussian",
        # can be NULL, but in R (use 0 in R instead of NULL for v0.7.0)
        block_length=3,
        margins="empirical",
        B=100,
        type_aggregation="mean",
        centers=2,
        type_clustering="kmeans",
        cl=1,
        date_formatting="original",
        seed=123,
    ):

        super().__init__(
            h=h,
            level=level,
            seed=seed,
        )

        self.lags = lags
        self.nb_hidden = nb_hidden
        self.nodes_sim = nodes_sim
        self.activation = activation
        self.a = a
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.dropout = dropout
        self.type_pi = type_pi
        self.block_length = block_length
        self.margins = margins
        self.B = B
        self.type_aggregation = type_aggregation
        # can be NULL, but in R (use 0 in R instead of NULL for v0.7.0)
        self.centers = centers
        self.type_clustering = type_clustering
        self.cl = cl
        self.date_formatting = date_formatting
        self.seed = seed
        self.input_df = None
        self.type_input = "multivariate"

        self.fcast_ = None
        self.averages_ = None
        self.ranges_ = None
        self.output_dates_ = []
        self.mean_ = None
        self.lower_ = None
        self.upper_ = None
        self.result_dfs_ = None
        self.sims_ = None
        self.xreg_ = None

    def forecast(self, df, xreg=None):
        """Forecasting method from `Ridge2Regressor` class

        Parameters:

            df: a data frame;
                a data frame containing the input time series (see example)

            xreg: a numpy array or a data frame;
                external regressors

        """

        # get input dates, output dates, number of series, series names, etc.
        self.init_forecasting_params(df)

        # obtain time series object -----
        self.format_input()

        self.get_forecast("ridge2")

        # result -----
        try: 
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
        except Exception:
            # result -----
            (
                self.averages_,
                self.ranges_,
                _,
            ) = uv.format_univariate_forecast(
                date_formatting=self.date_formatting,
                output_dates=self.output_dates_,
                horizon=self.h,
                fcast=self.fcast_,
            )

        self.mean_ = np.asarray(self.fcast_.rx2["mean"])
        self.lower_ = np.asarray(self.fcast_.rx2["lower"])
        self.upper_ = np.asarray(self.fcast_.rx2["upper"])

        try: 
            self.result_dfs_ = tuple(
                umv.compute_result_df(self.averages_[i], self.ranges_[i])
                for i in range(self.n_series)
            )
        except Exception: 
            self.result_df_ = umv.compute_result_df(self.averages_, self.ranges_)

        if self.type_pi in (
            "bootstrap",
            "blockbootstrap",
            "movingblockbootstrap",
            "rvinecopula",
        ):
            self.sims_ = tuple(
                np.asarray(self.fcast_.rx2["sims"][i]) for i in range(self.B)
            )

        return self
