# EAT

_EAT (ETS, auto.arima, Theta) model_

<span style="float:right;">[[source]](https://github.com/Techtonique/ahead/ahead/EAT/EAT.py#L46)</span>

### EAT


```python
ahead.EAT.EAT.EAT(
    h=5,
    level=95,
    weights=[0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
    type_pi="E",
    date_formatting="original",
)
```


Combinations of ETS (exponential smoothing), auto.arima and Theta models

Parameters:

    h: an integer;
        forecasting horizon

    level: an integer;
        Confidence level for prediction intervals

    weights: a list;
        coefficients assigned to each method in the ensemble

    type_pi: a string;
        Type of prediction interval (currently "gaussian",
        ETS: "E", Arima: "A" or Theta: "T")

    date_formatting: a string;
        Currently:
        - "original": yyyy-mm-dd
        - "ms": milliseconds

Attributes:

    fcast_: an object;
        raw result from fitting R's `ahead::eatf` through `rpy2`

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

Examples:

    ```
        import pandas as pd
        from ahead import EAT

        # Data frame containing the time series
        dataset = {
        'date' : ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01'],
        'value' : [34, 30, 35.6, 33.3, 38.1]}

        df = pd.DataFrame(dataset).set_index('date')
        print(df)

        # univariate time series forecasting
        e1 = EAT(h = 5) # default, equal weights for each model=[1/3, 1/3, 1/3]
        e1.forecast(df)
        print(e1.result_df_)
    ```


----

<span style="float:right;">[[source]](https://github.com/Techtonique/ahead/ahead/EAT/EAT.py#L144)</span>

### forecast


```python
EAT.forecast(df)
```


Forecasting method from `EAT` class

Parameters:

    df: a data frame;
        a data frame containing the input time series (see example)


----

