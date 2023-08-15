from .. import config

def plot(obj, selected_series = None):
    if selected_series is not None:        
        return(config.PLOT_AHEAD(obj.fcast_, selected_series))
    else: 
        return(config.PLOT_BASE(obj.fcast_))