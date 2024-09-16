import numpy as np
import pandas as pd
from difflib import SequenceMatcher


# compute input dates from data frame's index
def compute_input_dates(df):

    input_dates = df.index.values

    frequency = pd.infer_freq(pd.DatetimeIndex(input_dates))

    input_dates = pd.date_range(
        start=input_dates[0], periods=len(input_dates), freq=frequency
    ).values.tolist()

    df_input_dates = pd.DataFrame({"date": input_dates})

    input_dates = pd.to_datetime(df_input_dates["date"]).dt.date

    return input_dates


# compute output dates from data frame's index
def compute_output_dates(df, horizon):
    input_dates = df.index.values

    if input_dates[0] == 0:
        input_dates = pd.date_range(
            start=pd.Timestamp.today().strftime("%Y-%m-%d"), periods=horizon
        )

    # print(f"\n in nnetsauce.utils.timeseries 1: {input_dates} \n")

    frequency = pd.infer_freq(pd.DatetimeIndex(input_dates))

    # print(f"\n in nnetsauce.utils.timeseries 2: {frequency} \n")

    output_dates = np.delete(
        pd.date_range(
            start=input_dates[-1], periods=horizon + 1, freq=frequency
        ).values,
        0,
    ).tolist()

    # print(f"\n in nnetsauce.utils.timeseries 3: {output_dates} \n")

    df_output_dates = pd.DataFrame({"date": output_dates})

    output_dates = pd.to_datetime(df_output_dates["date"]).dt.date

    return output_dates, frequency


def compute_result_df(averages, ranges):
    try:
        pred_mean = pd.Series(dict(averages)).to_frame("mean")
    except Exception:
        pred_mean = pd.Series(averages).to_frame("mean")
    pred_ci = pd.DataFrame(
        ranges, columns=["date", "lower", "upper"]
    ).set_index("date")
    return pd.concat([pred_mean, pred_ci], axis=1)


def get_closest_str(input_str, list_choices):
    scores = np.asarray(
        [
            SequenceMatcher(None, a=input_str, b=elt).ratio()
            for elt in list_choices
        ]
    )
    return list_choices[np.where(scores == np.max(scores))[0][0]]


def get_frequency(input_str):
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    """https://otexts.com/fpp2/ts-objects.html#frequency-of-a-time-series
    Data 	frequency
    Annual 	        1
    Quarterly 	    4
    Monthly 	    12
    Weekly 	        52
    """
    frequency_choices = {
        "A": 1,
        "Y": 1,
        "BA": 1,
        "BY": 1,
        "AS": 1,
        "YS": 1,
        "BAS": 1,
        "BYS": 1,
        "Q": 4,
        "BQ": 4,
        "QS": 4,
        "BQS": 4,
        "M": 12,
        "BM": 12,
        "CBM": 12,
        "MS": 12,
        "BMS": 12,
        "CBMS": 12,
        "W": 52,
        "B": 365,
        "C": 365,
        "D": 365,
    }

    try:

        return frequency_choices[input_str]

    except:

        list_frequency_choices = list(frequency_choices.keys())

        closest_str = get_closest_str(
            input_str=input_str, list_choices=list_frequency_choices
        )

        return frequency_choices[closest_str]
