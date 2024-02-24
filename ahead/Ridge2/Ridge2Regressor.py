import matplotlib.pyplot as plt
import numpy as np
import rpy2.robjects.conversion as cv
from rpy2.robjects import (
    default_converter,
    FloatVector,
    ListVector,
    numpy2ri,
    r,
)

from ..Base import Base
from ..utils import multivariate as mv
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
            "bootstrap", (circular) "blockbootstrap", "movingblockbootstrap", or "rvinecopula")

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
        lags=1,
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

        self.input_df = df                
        self.n_series = n_series
        self.series_names = df.columns
        n_series = len(self.series_names)

        # obtain dates 'forecast' -----

        output_dates, frequency = umv.compute_output_dates(
            self.input_df, self.h
        )

        # obtain time series forecast -----

        y = mv.compute_y_mts(self.input_df, frequency)

        if xreg is None:

            self.fcast_ = config.AHEAD_PACKAGE.ridge2f(
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
                dropout=self.dropout,
                type_pi=self.type_pi,
                margins=self.margins,
                # can be NULL, but in R (use 0 in R instead of NULL for v0.7.0)
                block_length=self.block_length,
                B=self.B,
                type_aggregation=self.type_aggregation,
                # can be NULL, but in R (use 0 in R instead of NULL for v0.7.0)
                centers=self.centers,
                type_clustering=self.type_clustering,
                cl=self.cl,
                seed=self.seed,
            )

        else:  # xreg is not None:

            try:
                self.xreg_ = xreg.values
            except:
                self.xreg_ = config.DEEP_COPY(xreg)

            is_matrix_xreg = len(self.xreg_.shape) > 1

            numpy2ri.activate()

            xreg_ = (
                r.matrix(
                    FloatVector(self.xreg_.flatten()),
                    byrow=True,
                    nrow=self.xreg_.shape[0],
                    ncol=self.xreg_.shape[1],
                )
                if is_matrix_xreg
                else r.matrix(
                    FloatVector(self.xreg_.flatten()),
                    byrow=True,
                    nrow=self.xreg_.shape[0],
                    ncol=1,
                )
            )

            self.fcast_ = config.AHEAD_PACKAGE.ridge2f(
                y,
                xreg=xreg_,
                h=self.h,
                level=self.level,
                lags=self.lags,
                nb_hidden=self.nb_hidden,
                nodes_sim=self.nodes_sim,
                activ=self.activation,
                a=self.a,
                lambda_1=self.lambda_1,
                lambda_2=self.lambda_2,
                dropout=self.dropout,
                type_pi=self.type_pi,
                margins=self.margins,
                # can be NULL, but in R (use 0 in R instead of NULL for v0.7.0)
                block_length=self.block_length,
                B=self.B,
                type_aggregation=self.type_aggregation,
                # can be NULL, but in R (use 0 in R instead of NULL for v0.7.0)
                centers=self.centers,
                type_clustering=self.type_clustering,
                cl=self.cl,
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

        self.mean_ = np.asarray(self.fcast_.rx2["mean"])
        self.lower_ = np.asarray(self.fcast_.rx2["lower"])
        self.upper_ = np.asarray(self.fcast_.rx2["upper"])

        self.result_dfs_ = tuple(
            umv.compute_result_df(self.averages_[i], self.ranges_[i])
            for i in range(n_series)
        )

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

    def plot(self, series, type_axis="dates", type_plot="pi"):
        """Plot time series forecast

        Parameters:

            series: {integer} or {string}
                series index or name
        """

        assert all(
            [
                self.mean_ is not None,
                self.lower_ is not None,
                self.upper_ is not None,
                self.output_dates_ is not None,
            ]
        ), "model forecasting must be obtained first (with predict)"

        if isinstance(series, str):
            assert (
                series in self.series_names
            ), f"series {series} doesn't exist in the input dataset"
            series_idx = self.df_.columns.get_loc(series)
        else:
            assert isinstance(series, int) and (
                0 <= series < self.n_series
            ), f"check series index (< {self.n_series})"
            series_idx = series

        y_all = list(self.df_.iloc[:, series_idx]) + list(
            self.mean_.iloc[:, series_idx]
        )
        y_test = list(self.mean_.iloc[:, series_idx])
        n_points_all = len(y_all)
        n_points_train = self.df_.shape[0]

        if type_axis == "numeric":
            x_all = [i for i in range(n_points_all)]
            x_test = [i for i in range(n_points_train, n_points_all)]

        if type_axis == "dates":  # use dates
            x_all = np.concatenate(
                (self.input_dates.values, self.output_dates_.values), axis=None
            )
            x_test = self.output_dates_.values

        if type_plot == "pi":
            fig, ax = plt.subplots()
            ax.plot(x_all, y_all, "-")
            ax.plot(x_test, y_test, "-", color="orange")
            ax.fill_between(
                x_test,
                self.lower_.iloc[:, series_idx],
                self.upper_.iloc[:, series_idx],
                alpha=0.2,
                color="orange",
            )
            plt.title(
                f"prediction intervals for {self.replications} simulations of {series}",
                loc="left",
                fontsize=12,
                fontweight=0,
                color="black",
            )
            plt.show()

        if type_plot == "spaghetti":
            palette = plt.get_cmap("Set1")
            sims_ix = getsims(self.sims_, series_idx)
            plt.plot(x_all, y_all, "-")
            for col_ix in range(
                sims_ix.shape[1]
            ):  # avoid this when there are thousands of simulations
                plt.plot(
                    x_test,
                    sims_ix[:, col_ix],
                    "-",
                    color=palette(col_ix),
                    linewidth=1,
                    alpha=0.9,
                )
            plt.plot(x_all, y_all, "-", color="black")
            plt.plot(x_test, y_test, "-", color="blue")
            # Add titles
            plt.title(
                f"{self.replications} simulations of {series}",
                loc="left",
                fontsize=12,
                fontweight=0,
                color="black",
            )
            plt.xlabel("Time")
            plt.ylabel("Values")
            # Show the graph
            plt.show()
