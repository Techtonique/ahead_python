# ahead | <a class="github-button" href="https://github.com/Techtonique/ahead_python/stargazers" data-color-scheme="no-preference: light; light: light; dark: dark;" data-size="large" aria-label="Star the ahead /the ahead  on GitHub">Star</a>

![PyPI](https://img.shields.io/pypi/v/ahead) [![PyPI - License](https://img.shields.io/pypi/l/ahead)](https://github.com/Techtonique/ahead_python/blob/main/LICENSE) [![Downloads](https://pepy.tech/badge/ahead)](https://pepy.tech/project/ahead) [![Last Commit](https://img.shields.io/github/last-commit/Techtonique/ahead_python)](https://github.com/Techtonique/ahead_python)


Welcome to __ahead__'s (Python version) website. 

`ahead` is a package for univariate and multivariate **time series forecasting**. The Python version is built on top of [the R package](https://techtonique.github.io/ahead/) with the same name. __ahead__'s source code is [available on GitHub](https://github.com/Techtonique/ahead_python).

Currently, 4 forecasting methods are implemented in the Python package:

- `DynamicRegressor`: **univariate** time series forecasting method adapted from [`forecast::nnetar`](https://otexts.com/fpp2/nnetar.html#neural-network-autoregression). The Python implementation contains only the [automatic version](https://thierrymoudiki.github.io/blog/2021/10/22/r/misc/ahead-ridge).
- `EAT`: **univariate** time series forecasting method based on combinations of R's `forecast::ets`, `forecast::auto.arima`, and `forecast::thetaf`
- `Ridge2Regressor`: **multivariate** time series forecasting method, based on __quasi-randomized networks__ and presented in [this paper](https://www.mdpi.com/2227-9091/6/1/22)
- `VAR`: **multivariate** time series forecasting method using Vector AutoRegressive model (VAR, mostly here for benchmarking purpose)

Looking for a specific function? You can also use the __search__ function available in the navigation bar.

## Installing

- From Pypi, stable version:

```bash
pip install ahead
```

- From Github, for the development version: 

```bash
pip install git+https://github.com/Techtonique/ahead.git
```

## Quickstart 

### Univariate time series

```python
import pandas as pd
from ahead import DynamicRegressor

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

### Multivariate time series

```python
import pandas as pd
from ahead import Ridge2Regressor

# Data frame containing the (3) time series
dataset = {
 'date' : ['2001-01-01', '2002-01-01', '2003-01-01', '2004-01-01', '2005-01-01'],
 'series1' : [34, 30, 35.6, 33.3, 38.1],    
 'series2' : [4, 5.5, 5.6, 6.3, 5.1],
 'series3' : [100, 100.5, 100.6, 100.2, 100.1]}
df = pd.DataFrame(dataset).set_index('date')

# multivariate time series forecasting 
r1 = Ridge2Regressor(h = 5)
r1.forecast(df)
print(r1.result_dfs_)
```


## Documentation

### For univariate models 

- For [DynamicRegressor](documentation/dynamicregressor.md)

- For [EAT](documentation/eat.md)

### For multivariate models 

- For [Ridge2Regressor](documentation/ridge2regressor.md)

- For [VAR](documentation/var.md)


## Contributing

Want to contribute to __ahead__'s development on Github, [read this](CONTRIBUTING.md)!

<script async defer src="https://buttons.github.io/buttons.js"></script>