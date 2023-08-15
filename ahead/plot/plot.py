from .. import config

def plot(obj, **kwargs):
    if "selected_series" in kwargs:
        config.PLOT_AHEAD(obj.fcast_, selected_series)
    else: 
        config.PLOT_BASE(obj.fcast_)