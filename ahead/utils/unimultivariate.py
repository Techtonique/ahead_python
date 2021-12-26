import numpy as np
import pandas as pd

from fuzzywuzzy import process

def compute_output_dates(df, horizon):

    input_dates = df.index.values

    frequency = pd.infer_freq(pd.DatetimeIndex(input_dates))
    output_dates = np.delete(
        pd.date_range(
            start=input_dates[-1], periods=horizon + 1, freq=frequency
        ).values,
        0,
    ).tolist()

    df_output_dates = pd.DataFrame({"date": output_dates})
    output_dates = pd.to_datetime(df_output_dates["date"]).dt.date

    return output_dates, frequency


def compute_result_df(averages, ranges):
    pred_mean = pd.Series(dict(averages)).to_frame("mean")
    pred_ci = pd.DataFrame(
        ranges, columns=["date", "lower", "upper"]
    ).set_index("date")
    return pd.concat([pred_mean, pred_ci], axis=1)


def get_frequency(input_str):
    # https://pandas.pydata.org/pandas-docs/stable/user_guide/timeseries.html#offset-aliases
    """ https://otexts.com/fpp2/ts-objects.html#frequency-of-a-time-series
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
        
        closest_str=process.extractOne(input_str, list(frequency_choices.keys()))[0]

        return frequency_choices[closest_str]