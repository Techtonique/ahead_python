import numpy as np
import matplotlib.pyplot as plt

from subprocess import Popen, PIPE
from .. import config


class Base(object):

    def __init__(self, h=5, level=95, date_formatting="ms", seed=123):

        self.h = h
        self.level = level
        self.date_formatting = date_formatting
        self.seed = seed
        self.series_names = None
        self.n_series = None
        self.B = None
        self.input_df = None

        self.mean_ = None
        self.lower_ = None
        self.upper_ = None
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

    def getsims(self, input_tuple, ix):
        n_sims = len(input_tuple)
        res = [input_tuple[i].iloc[:, ix].values for i in range(n_sims)]
        return np.asarray(res).T

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
            x_all = np.concatenate(
                (self.input_df.index.values, self.output_dates_), axis=None
            )
            x_test = self.output_dates_

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
