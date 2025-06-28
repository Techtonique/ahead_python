"""Top-level package for ahead."""

__author__ = """T. Moudiki"""
__email__ = "thierry.moudiki@gmail.com"
__version__ = "0.10.0"

from .ARMAGARCH import ArmaGarch
from .Basic import BasicForecaster
from .DynamicRegressor import DynamicRegressor
from .EAT import EAT
from .FitForecast import FitForecaster
from .Ridge2 import Ridge2Regressor
from .VAR import VAR
from .MLARCH import MLARCH


__all__ = [
    "ArmaGarch",
    "BasicForecaster",
    "DynamicRegressor",
    "EAT",
    "FitForecaster",
    "Ridge2Regressor",
    "VAR",
    "MLARCH"
]
