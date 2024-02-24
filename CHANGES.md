# version 0.10.0

- Naming series in input data frame
- Plot method for all the objects 
- More about the code: begin refactoring and DRYing --> Base class 

# version 0.9.0

- Align with R version
- Progress bars for bootstrapping (independent, circular block, moving block)
- See also [https://github.com/Techtonique/ahead/blob/main/NEWS.md](https://github.com/Techtonique/ahead/blob/main/NEWS.md) 

# version 0.8.2

- plot ridge2

# version 0.8.1

- Align with R version (see [https://github.com/Techtonique/ahead/blob/main/NEWS.md#version-070](https://github.com/Techtonique/ahead/blob/main/NEWS.md#version-070) and [https://github.com/Techtonique/ahead/blob/main/NEWS.md#version-080](https://github.com/Techtonique/ahead/blob/main/NEWS.md#version-080)) as much as possible
- moving block bootstrap in `ridge2f`, `basicf`, in addition to circular block bootstrap from 0.6.2
- adjust R-Vine copulas on residuals for `ridge2f` simulation (with empirical and Gaussian marginals)

# version 0.6.2

- Add Block Bootstrap to `ridge2f` and `basicf`
- Add external regressors to `ridge2f`
- Add clustering with _K-Means_ and hierarchical clustering to `ridge2f`
- Install R or rpy2 if necessary (? weird)
- Refactor code for `rpy2` and R imports

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