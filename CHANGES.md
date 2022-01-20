# version 0.5.0

- add dropout regularization to `Ridge2Regressor`
- parallel execution for type_pi == bootstrap in `Ridge2Regressor` (done in R /!\, experimental)

# version 0.4.2

- new attributes mean, lower bound, upper bound forecast as numpy arrays

# version 0.4.1

- use `get_frequency` to get series frequency as a number
- create a function `get_tscv_indices` for getting time series cross-validation indices