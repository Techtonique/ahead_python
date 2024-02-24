import os
import numpy as np
import pandas as pd

import rpy2.robjects as robjects
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector
from datetime import datetime
from rpy2.robjects.vectors import StrVector
from .unimultivariate import get_frequency

base = importr("base")
stats = importr("stats")


def compute_y_ts(df, df_frequency):

    input_series = df.to_numpy()

    ts = stats.ts(
        FloatVector(input_series.flatten()),
        frequency=get_frequency(df_frequency),
    )

    return ts


def format_univariate_forecast(date_formatting, output_dates, horizon, fcast):

    if date_formatting == "original":
        output_dates_ = [
            datetime.strftime(output_dates[i], "%Y-%m-%d")
            for i in range(horizon)
        ]

    if date_formatting == "ms":
        output_dates_ = [
            int(
                datetime.strptime(str(output_dates[i]), "%Y-%m-%d").timestamp()
                * 1000
            )
            for i in range(horizon)
        ]

    averages = [
        [output_dates_[i], fcast.rx2["mean"][i]] for i in range(horizon)
    ]
    ranges = [
        [output_dates_[i], fcast.rx2["lower"][i], fcast.rx2["upper"][i]]
        for i in range(horizon)
    ]

    return averages, ranges, output_dates_
