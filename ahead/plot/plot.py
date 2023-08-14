from .. import config

def plot(obj, **kwargs):
    try:
        config.GRAPHICS.plot(obj)
    except: 
        config.PLOT(obj, **kwargs)