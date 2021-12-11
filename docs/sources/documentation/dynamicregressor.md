# DynamicRegressor

_Dynamic Regression model adapted from R's forecast::nnetar_

<span style="float:right;">[[source]](https://github.com/Techtonique/ahead/ahead/DynamicRegressor/DynamicRegressor.py#L37)</span>

### DynamicRegressor


```python
ahead.DynamicRegressor.DynamicRegressor.DynamicRegressor(
    h=5, level=95, type_pi="E", date_formatting="original"
)
```


Dynamic Regression Model adapted from R's `forecast::nnetar`

Parameters:
   
    h: an integer;
        forecasting horizon
    
    level: an integer;
        Confidence level for prediction intervals

    type_pi: a string;
        Type of prediction interval (currently "gaussian", 
        ETS: "E", Arima: "A" or Theta: "T")

    date_formatting: a string;
        Currently: 
        - "original": yyyy-mm-dd
        - "ms": milliseconds        

Attributes:
   
    fcast_: an object;
        raw result from fitting R's `ahead.dynrmf`
    
    averages_: a list;
        mean forecast in a list

    ranges_: a list;
        lower and upper prediction intervals in a list

    output_dates_: a list;
        a list of output dates (associated to forecast)

    result_df_: a data frame;
        contains 3 columns, mean forecast, lower + upper 
        prediction intervals, and a date index          

Examples:
   
    ```
        # Data frame containing the time series 
        dataset = {
        'date' : ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01'],
        'value' : [34, 30, 35.6, 33.3, 38.1]}

        df = pd.DataFrame(dataset).set_index('date')
        print(df)

        # univariate time series forecasting 
        d1 = DynamicRegressor(h = 5)
        d1.forecast(df)
        print(d1.result_df_)
    ```


----

<span style="float:right;">[[source]](https://github.com/Techtonique/ahead/ahead/DynamicRegressor/DynamicRegressor.py#L108)</span>

### forecast


```python
DynamicRegressor.forecast(df)
```


Forecasting method from `DynamicRegressor` class

Parameters:

    df: a data frame;
        a data frame containing the time series (see example)                


----

