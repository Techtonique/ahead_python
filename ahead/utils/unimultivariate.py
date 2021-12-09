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

def compute_result_df(averages, ranges):
    pred_mean = pd.Series(dict(averages)).to_frame('mean')
    pred_ci = pd.DataFrame(ranges, columns=['date', 'lower', 'upper']).set_index('date')
    return pd.concat([pred_mean, pred_ci], axis=1)