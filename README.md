ahead 
===============================


![PyPI](https://img.shields.io/pypi/v/ahead) [![PyPI - License](https://img.shields.io/pypi/l/ahead)](https://github.com/Techtonique/ahead_python/blob/main/LICENSE) [![PyPI Downloads](https://pepy.tech/badge/ahead)](https://pepy.tech/project/ahead) [![Conda Downloads](https://img.shields.io/conda/dn/conda-forge/ahead_python.svg)](https://anaconda.org/conda-forge/ahead_python)
[![HitCount](https://hits.dwyl.com/Techtonique/ahead_python.svg?style=flat-square)](http://hits.dwyl.com/Techtonique/ahead_python)
[![Documentation](https://img.shields.io/badge/documentation-is_here-green)](https://techtonique.github.io/ahead_python/)


Welcome to __ahead__ (Python version; the R version is [here](https://github.com/Techtonique/ahead)). 

`ahead` is a package for univariate and multivariate **time series forecasting**, with uncertainty quantification. The Python version is built on top of [the R package](https://techtonique.github.io/ahead/) with the same name. __ahead__'s source code is [available on GitHub](https://github.com/Techtonique/ahead_python).

Currently, 6 forecasting methods are implemented in the Python package:

- `DynamicRegressor`: **univariate** time series forecasting method adapted from [`forecast::nnetar`](https://otexts.com/fpp2/nnetar.html#neural-network-autoregression). 
The Python implementation contains only the [automatic version](https://thierrymoudiki.github.io/blog/2021/10/22/r/misc/ahead-ridge).
- `EAT`: **univariate** time series forecasting method based on combinations of R's `forecast::ets`, `forecast::auto.arima`, and `forecast::thetaf`
- `ArmaGarch`: **univariate** forecasting simulations of an ARMA(1, 1)-GARCH(1, 1)
- `BasicForecaster`: **multivariate** time series forecasting methods; mean, median and random walk
- `Ridge2Regressor`: **multivariate** time series forecasting method, based on __quasi-randomized networks__ and presented in [this paper](https://www.mdpi.com/2227-9091/6/1/22)
- `VAR`: **multivariate** time series forecasting method using Vector AutoRegressive model (VAR, mostly here for benchmarking purpose)

## Installing

- From Pypi, stable version:

```bash
pip install ahead --verbose
```

- From Github, for the development version: 

```bash
pip install git+https://github.com/Techtonique/ahead_python.git --verbose
```

## Quickstart 

### Univariate time series

```python
import pandas as pd
from ahead import DynamicRegressor # might take some time, but ONLY the 1st time it's called

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
from ahead import Ridge2Regressor # might take some time, but ONLY the 1st time it's called

# Data frame containing the (3) time series
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

## Contributing

Want to contribute to __ahead__'s development on Github, [read this](CONTRIBUTING.md)!

## License

[BSD 3-Clause](LICENSE) © Thierry Moudiki, 2021. 
