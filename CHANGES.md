# version 0.7.0

- Install R or rpy2 if necessary (?)
- Add Block Bootstrap to `ridge2f`
- Add external regressors to `ridge2f`

# version 0.6.1

- Reduce number of required packages depencies
- Fix rpy2 requirement (version 3.4.5)
- Requires Python version >= 3.9

# version 0.5.0

- add dropout regularization to `Ridge2Regressor`
- parallel execution for `type_pi == bootstrap` in `Ridge2Regressor` (done in R /!\, experimental)
- ARMA(1, 1)-GARCH(1, 1) in Python 

# version 0.4.2

- new attributes mean, lower bound, upper bound forecast as numpy arrays

# version 0.4.1

- use `get_frequency` to get series frequency as a number
- create a function `get_tscv_indices` for getting time series cross-validation indices