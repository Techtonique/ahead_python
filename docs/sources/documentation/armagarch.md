# ARMA(1, 1)-GARCH(1, 1)

_ARMA(1, 1)-GARCH(1, 1) model_

<span style="float:right;">[[source]](https://github.com/Techtonique/ahead/ahead/ARMAGARCH/ArmaGarch.py#L46)</span>

### ArmaGarch


```python
ahead.ARMAGARCH.ArmaGarch.ArmaGarch(
    h=5, level=95, B=250, cl=1, dist="student", seed=123, date_formatting="original"
)
```


ARMA(1, 1)-GARCH(1, 1) forecasting (with simulation)

Parameters:

    h: an integer;
        forecasting horizon

    level: an integer;
        Confidence level for prediction intervals

    B: an integer;
        number of simulations for R's `stats::arima.sim`

    cl: an integer;
        the number of clusters for parallel execution (done in R /!\)

    dist: a string;
        distribution of innovations ("student" or "gaussian")

    seed: an integer;
        reproducibility seed                

    date_formatting: a string;
        Currently:
        - "original": yyyy-mm-dd
        - "ms": milliseconds

Attributes:

    fcast_: an object;
        raw result from fitting R's `ahead::armagarchf` through `rpy2`

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

    sims_: a numpy array
        forecasting simulations


----

<span style="float:right;">[[source]](https://github.com/Techtonique/ahead/ahead/ARMAGARCH/ArmaGarch.py#L135)</span>

### forecast


```python
ArmaGarch.forecast(df)
```


Forecasting method from `ArmaGarch` class

Parameters:

    df: a data frame;
        a data frame containing the input time series (see example)


----

