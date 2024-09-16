import numpy as np
import matplotlib.pyplot as plt

from rpy2.robjects import numpy2ri, r
from rpy2.robjects.vectors import FloatVector

from subprocess import Popen, PIPE
from .. import config
from ..utils.univariate import compute_y_ts
from ..utils.multivariate import compute_y_mts
from ..utils.unimultivariate import compute_input_dates, compute_output_dates


class Base(object):

    def __init__(self, h=5, level=95, date_formatting="ms", seed=123):

        self.h = h
        self.level = level
        self.date_formatting = date_formatting
        self.seed = seed
        self.frequency = None
        self.series_names = None
        self.n_series = None
        self.type_input = "univariate"  # (or "multivariate")
        self.B = None
        self.input_df = None
        self.input_dates = None
        self.method = None
        self.weights = None

        self.input_ts_ = None  # input time series
        self.mean_ = None
        self.lower_ = None
        self.upper_ = None
        self.sims_ = None
        self.output_dates_ = None
        self.result_dfs_ = None

        R_IS_INSTALLED = False

        try:
            proc = Popen(["which", "R"], stdout=PIPE, stderr=PIPE)
            R_IS_INSTALLED = proc.wait() == 0
        except Exception as e:
            pass

        if not R_IS_INSTALLED:
            raise ImportError("R is not installed! \n" + config.USAGE_MESSAGE)

    def format_input(self):
        if self.input_df.shape[1] > 0:
            self.input_ts_ = compute_y_mts(self.input_df, self.frequency)
        else:
            self.input_ts_ = compute_y_ts(self.input_df, self.frequency)

    def init_forecasting_params(self, df):
        self.input_df = df
        self.series_names = df.columns
        self.n_series = len(self.series_names)
        self.input_dates = compute_input_dates(df)
        self.type_input = "multivariate" if len(df.shape) > 0 else "univariate"
        self.output_dates_, self.frequency = compute_output_dates(df, self.h)

    def getsims(self, input_tuple, ix):
        n_sims = len(input_tuple)
        res = [input_tuple[i].iloc[:, ix].values for i in range(n_sims)]
        return np.asarray(res).T

    def get_forecast(self, method=None, xreg=None):

        if method != None:
            self.method = method

        if self.method == "armagarch":
            self.fcast_ = config.AHEAD_PACKAGE.armagarchf(
                y=self.input_ts_,
                h=self.h,
                level=self.level,
                B=self.B,
                cl=self.cl,
                dist=self.dist,
                seed=self.seed,
            )

        if self.method in ("mean", "median", "rw"):
            self.fcast_ = config.AHEAD_PACKAGE.basicf(
                self.input_ts_,
                h=self.h,
                level=self.level,
                method=self.method,
                type_pi=self.type_pi,
                block_length=self.block_length,
                B=self.B,
                seed=self.seed,
            )

        if self.method == "dynrm":
            self.fcast_ = config.AHEAD_PACKAGE.dynrmf(
                y=self.input_ts_,
                h=self.h,
                level=self.level,
                type_pi=self.type_pi,
            )

        if self.method == "eat":
            self.fcast_ = config.AHEAD_PACKAGE.eatf(
                y=self.input_ts_,
                h=self.h,
                level=self.level,
                type_pi=self.type_pi,
                weights=config.FLOATVECTOR(self.weights),
            )

        if self.method == "ridge2":
            if xreg is None:

                self.fcast_ = config.AHEAD_PACKAGE.ridge2f(
                    self.input_ts_,
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
                    self.input_ts_,
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

        if self.method == "var":
            self.fcast_ = config.AHEAD_PACKAGE.varf(
                self.input_ts_,
                h=self.h,
                level=self.level,
                lags=self.lags,
                type_VAR=self.type_VAR,
            )

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
        ), "model forecasting must be obtained first (with `forecast` method)"

        if isinstance(series, str):
            assert (
                series in self.series_names
            ), f"series {series} doesn't exist in the input dataset"
            series_idx = self.input_df.columns.get_loc(series)
        else:
            assert isinstance(series, int) and (
                0 <= series < self.n_series
            ), f"check series index (< {self.n_series})"
            series_idx = series

        y_all = list(self.input_df.iloc[:, series_idx]) + list(
            self.result_dfs_[series_idx]["mean"].values
        )

        y_test = list(self.result_dfs_[series_idx]["mean"].values)
        n_points_all = len(y_all)
        n_points_train = self.input_df.shape[0]

        if type_axis == "numeric":
            x_all = [i for i in range(n_points_all)]
            x_test = [i for i in range(n_points_train, n_points_all)]

        if type_axis == "dates":  # use dates
            x_train = [date.strftime("%Y-%m-%d") for date in self.input_dates]
            x_test = [date.strftime("%Y-%m-%d") for date in self.output_dates_]
            x_all = np.concatenate((x_train, x_test), axis=None)

        if type_plot == "pi":
            fig, ax = plt.subplots()
            ax.plot(x_all, y_all, "-")
            ax.plot(x_test, y_test, "-", color="orange")
            ax.fill_between(
                x_test,
                self.result_dfs_[series_idx]["lower"].values,
                self.result_dfs_[series_idx]["upper"].values,
                alpha=0.2,
                color="orange",
            )
            plt.title(
                f"prediction intervals for {series}",
                loc="left",
                fontsize=12,
                fontweight=0,
                color="black",
            )
            plt.show()

        if type_plot == "spaghetti":
            palette = plt.get_cmap("Set1")
            sims_ix = self.getsims(self.sims_, series_idx)
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
                f"{self.B} simulations of {series}",
                loc="left",
                fontsize=12,
                fontweight=0,
                color="black",
            )
            plt.xlabel("Time")
            plt.ylabel("Values")
            # Show the graph
            plt.show()
