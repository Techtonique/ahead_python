# BasicForecaster

_Basic forecasting functions for multivariate time series (mean, median, random walk)_

<span style="float:right;">[[source]](https://github.com/Techtonique/ahead_python/blob/main/ahead/Basic/BasicForecaster.py#L45)</span>

### BasicForecaster


```python
ahead.Basic.BasicForecaster.BasicForecaster(
    h=5, level=95, method="mean", type_pi="gaussian", B=100, date_formatting="original", seed=123
)
```


Basic forecasting functions for multivariate time series (mean, median, random walk)

Parameters:

    h: an integer;
        forecasting horizon

    level: an integer;
        Confidence level for prediction intervals

    method: a string;
        Forecasting method, either "mean", "median", or random walk ("rw")    

    type_pi: a string;
        Type of prediction interval (currently "gaussian",
        or "bootstrap")

    B: an integer;
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

```python
import pandas as pd
from ahead import BasicForecaster

# Data frame containing the time series
dataset = {
'date' : ['2001-01-01', '2002-01-01', '2003-01-01', '2004-01-01', '2005-01-01'],
'series1' : [34, 30, 35.6, 33.3, 38.1],
'series2' : [4, 5.5, 5.6, 6.3, 5.1],
'series3' : [100, 100.5, 100.6, 100.2, 100.1]}
df = pd.DataFrame(dataset).set_index('date')
print(df)

# multivariate time series forecasting
r1 = BasicForecaster(h = 5)
r1.forecast(df)
print(r1.result_dfs_)
```


----

<span style="float:right;">[[source]](https://github.com/Techtonique/ahead_python/blob/main/ahead/Basic/BasicForecaster.py#L156)</span>

### forecast


```python
BasicForecaster.forecast(df)
```


Forecasting method from `BasicForecaster` class

Parameters:

    df: a data frame;
        a data frame containing the input time series (see example)


----

