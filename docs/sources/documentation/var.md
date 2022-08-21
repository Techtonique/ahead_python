# VAR

_Vector AutoRegressive model_

<span style="float:right;">[[source]](https://github.com/Techtonique/ahead_python/blob/main/ahead/VAR/VAR.py#L45)</span>

### VAR


```python
ahead.VAR.VAR.VAR(h=5, level=95, lags=1, type_VAR="none", date_formatting="original")
```


Vector AutoRegressive model

Parameters:

    h: an integer;
        forecasting horizon

    level: an integer;
        Confidence level for prediction intervals

    lags: an integer;
        the lag order

    type_VAR: a string;
        Type of deterministic regressors to include
        ("const", "trend", "both", "none")

    date_formatting: a string;
        Currently:
        - "original": yyyy-mm-dd
        - "ms": milliseconds

Attributes:

    fcast_: an object;
        raw result from fitting R's `ahead::varf` through `rpy2`

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

Examples:

```python
import pandas as pd
from ahead import VAR

# Data frame containing the time series
dataset = {
'date' : ['2001-01-01', '2002-01-01', '2003-01-01', '2004-01-01', '2005-01-01'],
'series1' : [34, 30, 35.6, 33.3, 38.1],
'series2' : [4, 5.5, 5.6, 6.3, 5.1],
'series3' : [100, 100.5, 100.6, 100.2, 100.1]}
df = pd.DataFrame(dataset).set_index('date')
print(df)

# multivariate time series forecasting
v1 = VAR(h = 5, date_formatting = "original", type_VAR="none")
v1.forecast(df)
print(v1.result_dfs_)
```


----

<span style="float:right;">[[source]](https://github.com/Techtonique/ahead_python/blob/main/ahead/VAR/VAR.py#L144)</span>

### forecast


```python
VAR.forecast(df)
```


Forecasting method from `VAR` class

Parameters:

    df: a data frame;
        a data frame containing the input time series (see example)


----

