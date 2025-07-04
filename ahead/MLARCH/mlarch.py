import numpy as np

from ..Base import Base
from ..utils import univariate as uv
from ..utils import unimultivariate as umv
from .. import config


class MLARCH(Base):
    """Conformalized Forecasting using Machine Learning (and statistical) models with ARCH effects

    mean_model: `forecast::auto.arima` (main series)
    model_residuals: `forecast::thetaf` (residuals)
    fit_func: `ahead::ridge` (volatility)

    

    Parameters:

        h: an integer;
            forecasting horizon

        level: an integer;
            Confidence level for prediction intervals

        B: an integer;
            Number of bootstrap replications for `type_pi == bootstrap`, "blockbootstrap",
            "movingblockbootstrap", or "rvinecopula"

        type_pi: a string;
            Type of conformal prediction interval ("surrogate", "bootstrap", "kde") for volatility modeling

        type_sim_conformalize: a string;
            Type of simulation for conformalization of standardized residuals ("block-bootstrap", "surrogate", "kde", "bootstrap", or "fitdistr")

        date_formatting: a string;
            Currently:
            - "original": yyyy-mm-dd
            - "ms": milliseconds

    Attributes:

        fcast_: an object;
            raw result from fitting R's `ahead::MLARCHf` through `rpy2`

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

    Examples:

    ```python
    import pandas as pd
    from ahead import MLARCH

    # Data frame containing the time series
    dataset = {
    'date' : ['2020-01-01', '2020-02-01', '2020-03-01', '2020-04-01', '2020-05-01'],
    'value' : [34, 30, 35.6, 33.3, 38.1]}

    df = pd.DataFrame(dataset).set_index('date')
    print(df)

    # univariate time series forecasting
    e1 = MLARCH(h = 5) 
    e1.forecast(df)
    print(e1.result_df_)
    ```

    """

    def __init__(
        self,
        h=5,
        level=95,
        B=100,
        type_pi="surrogate",
        type_sim_conformalize="block-bootstrap",
        seed=123,
        date_formatting="original",
    ):

        super().__init__(h=h, level=level)
        
        self.h = h
        self.level = level
        self.B = B
        self.type_pi = type_pi
        self.type_sim_conformalize = type_sim_conformalize  
        self.seed = seed
        self.date_formatting = date_formatting
        self.input_df = None
        self.type_input = "univariate"

        self.fcast_ = None
        self.averages_ = None
        self.ranges_ = None
        self.output_dates_ = []
        self.mean_ = []
        self.lower_ = []
        self.upper_ = []
        self.result_df_ = None

    def forecast(self, df):
        """Forecasting method from `MLARCH` class

        Parameters:

            df: a data frame;
                a data frame containing the input time series (see example)

        """

        # get input dates, output dates, number of series, series names, etc.
        self.init_forecasting_params(df)

        # obtain time series object -----
        self.format_input()

        self.get_forecast("MLARCH")

        print(f"MLARCH: {self.fcast_}")

        # result -----
        (
            self.averages_,
            self.ranges_,
            _,
        ) = uv.format_univariate_forecast(
            date_formatting=self.date_formatting,
            output_dates=self.output_dates_,
            horizon=self.h,
            fcast=self.fcast_,
        )

        self.mean_ = np.asarray(self.fcast_.rx2["mean"])
        self.lower_ = np.asarray(self.fcast_.rx2["lower"])
        self.upper_ = np.asarray(self.fcast_.rx2["upper"])

        self.result_df_ = umv.compute_result_df(self.averages_, self.ranges_)

        return self
