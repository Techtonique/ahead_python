import numpy as np

from ..Base import Base
from ..utils import univariate as uv
from ..utils import unimultivariate as umv
from .. import config
from rpy2.robjects import NULL as rNULL

class FitForecaster(Base):
    """ Fit and forecast time series with uncertainty quantification

    See https://techtonique.r-universe.dev/ahead/doc/manual.html#fitforecast  

    Examples:

    ```python
    url = "https://raw.githubusercontent.com/Techtonique/"
    url += "datasets/main/time_series/univariate/"
    url += "a10.csv"

    df = pd.read_csv(url)
    df.index = pd.DatetimeIndex(df.date) # must have
    df.drop(columns=['date'], inplace=True)

    # univariate ts forecasting 
    d1 = FitForecaster()

    print(d1)

    start = time()
    print(d1.fit_forecast(df))
    print(f"Elapsed: {time()-start} \n")

    print(f"after: {d1.mean_}")
    print(f"after: {d1.lower_}")
    print(f"after: {d1.upper_}")
    ```

    """

    def __init__(self, h=None, level=95, 
                 pct_train = 0.9,
                 pct_calibration = 0.5,                
                 B = 1000,
                 seed = 17223,                     
                 conformalize = False,
                 type_calibration = "splitconformal",
                 gap = 3,
                 agg = "mean",
                 vol = "constant",
                 type_sim = "kde",
                 date_formatting="original"):

        super().__init__(
            h=h,
            level=level,
        )

        self.pct_train = pct_train
        self.pct_calibration = pct_calibration
        self.B = B
        self.seed = seed                 
        self.conformalize = conformalize
        self.type_calibration = type_calibration
        self.gap = gap 
        self.agg = agg 
        self.vol = vol 
        self.type_sim = type_sim
        self.date_formatting = date_formatting
        self.input_df = None        

        self.fcast_ = None
        self.averages_ = None
        self.ranges_ = None
        self.output_dates_ = []
        self.mean_ = []
        self.lower_ = []
        self.upper_ = []
        self.result_df_ = None

    def fit_forecast(self, df, 
                     method="thetaf"):

        assert method in ("thetaf", "arima", "ets", "te", "tbats", "tslm", "dynrmf", "ridge2f", "naive", "snaive"), \
        'must have method in ("thetaf", "arima", "ets", "te", "tbats", "tslm", "dynrmf", "ridge2f", "naive", "snaive")'
        
        # keep it in this order
        h = None 
        if self.h is not None:
            h = self.h
        else: 
            self.h = np.floor(df.shape[0]*self.pct_calibration)

        # get input dates, output dates, number of series, series names, etc. 
        self.init_forecasting_params(df)        

        # obtain time series object -----
        self.format_input()

        self.method = method 
        
        self.fcast_ = config.AHEAD_PACKAGE.fitforecast(
        y=self.input_ts_,
        h=rNULL if h is None else h,            
        pct_train = self.pct_train,
        pct_calibration = self.pct_calibration,
        method = self.method,
        level = self.level,
        B = self.B,
        seed = self.seed,
        conformalize = self.conformalize,
        type_calibration = self.type_calibration,
        )

        #print(self.fcast_)

        #print(dir(self.fcast_))

        #print(self.fcast_.names)

        #print(self.fcast_.slots)

        # Extract the keys from self.fcast_
        for idx, key in enumerate(self.fcast_.names):
            try:
                # Access the R list element by name using rx2
                item = self.fcast_.rx2[key]                
                # Check if the item is not NULL
                if item != rNULL:
                    try: 
                        # Convert the R object to a NumPy array
                        if df.shape[1] > 1:
                            item_array = np.asarray(item)                    
                        else: 
                            item_array = np.asarray(item).ravel()                    
                    except: 
                        item_array = item                    
                    # Dynamically set the attribute in the Python object
                    setattr(self, f"{key}_", item_array)            
            except Exception as e:
                print(f"Error processing idx {idx}: {e}")
                continue
        
        return self 


