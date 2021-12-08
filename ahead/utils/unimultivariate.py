import numpy as np
import pandas as pd

def compute_output_dates(df, horizon): 

    input_dates = df.index.values 
        
    frequency = pd.infer_freq(pd.DatetimeIndex(input_dates))
    output_dates = np.delete(pd.date_range(start=input_dates[-1], 
            periods=horizon+1, freq=frequency).values, 0).tolist()  
        
    df_output_dates = pd.DataFrame({'date': output_dates})
    output_dates = pd.to_datetime(df_output_dates['date']).dt.date 

    return output_dates, frequency
