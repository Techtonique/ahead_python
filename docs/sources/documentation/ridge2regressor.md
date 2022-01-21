# Ridge2Regressor

_Random Vector functional link network model with 2 regularization parameters_

<span style="float:right;">[[source]](https://github.com/Techtonique/ahead/ahead/Ridge2/Ridge2Regressor.py#L45)</span>

### Ridge2Regressor


```python
ahead.Ridge2.Ridge2Regressor.Ridge2Regressor(
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
    B=100,
    cl=1,
    date_formatting="original",
    seed=123,
)
```


Random Vector functional link network model with 2 regularization parameters

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
        or "bootstrap")

    B: an integer;
        Number of bootstrap replications for `type_pi == bootstrap`

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

    ```
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


----

<span style="float:right;">[[source]](https://github.com/Techtonique/ahead/ahead/Ridge2/Ridge2Regressor.py#L198)</span>

### forecast


```python
Ridge2Regressor.forecast(df)
```


Forecasting method from `Ridge2Regressor` class

Parameters:

    df: a data frame;
        a data frame containing the input time series (see example)


----

