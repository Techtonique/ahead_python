from .. import config

def plot(obj, selected_series = None):

    print(f" \n obj.fcast_: {obj.fcast_} \n ")

    if selected_series is not None:
        try: 
            config.PLOT_AHEAD(obj.fcast_, selected_series)
        except:
            pass
    else: 
        config.PLOT_BASE(obj.fcast_)