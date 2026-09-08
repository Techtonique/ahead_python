"""Top-level package for ahead."""

__author__ = """T. Moudiki"""
__email__ = "thierry.moudiki@gmail.com"
__version__ = "0.38.2"

from .ARMAGARCH import ArmaGarch
from .Basic import BasicForecaster
from .DynamicRegressor import DynamicRegressor
from .EAT import EAT
from .FitForecast import FitForecaster
from .Ridge2 import Ridge2Regressor
from .VAR import VAR
from .MLARCH import MLARCH

# ahead/__init__.py, near the top
import shutil, sys

if shutil.which("Rscript") is None:
    sys.exit(
        "R is required by 'ahead' but wasn't found. "
        "Install it from https://cran.r-project.org and re-install this package."
    )

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
