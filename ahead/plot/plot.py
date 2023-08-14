from .. import config

def plot(obj, **kwargs):
    try:
        config.GRAPHICS.plot(obj.fcast_)
    except: 
        config.PLOT(obj.fcast_, **kwargs)