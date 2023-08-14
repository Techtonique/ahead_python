from .. import config

def plot(obj, **kwargs):
    try:
        config.PLOT_AHEAD(obj.fcast_, **kwargs)
    except: 
        config.PLOT_BASE(obj.fcast_, **kwargs)