"""Top-level package for ahead."""

__author__ = """T. Moudiki"""
__email__ = "thierry.moudiki@gmail.com"
__version__ = "0.38.5"

import shutil, sys
import subprocess

if shutil.which("Rscript") is None:
    sys.exit(
        "R is required by 'ahead' but wasn't found. "
        "Install it from https://cran.r-project.org and re-install this package."
    )

def _ensure_r_ahead_installed():
    if shutil.which("Rscript") is None:
        sys.exit(
            "R is required by the 'ahead' Python package but wasn't found.\n"
            "Install R from https://cran.r-project.org, then re-import ahead."
        )
    check_script = (
        "options(repos=c(techtonique='https://r-packages.techtonique.net', "
        "CRAN='https://cloud.r-project.org')); "
        "if (!requireNamespace('ahead', quietly=TRUE)) "
        "install.packages('ahead', dependencies=TRUE)"
    )
    result = subprocess.run(["Rscript", "-e", check_script])
    if result.returncode != 0:
        sys.exit(
            "Failed to install the R 'ahead' package automatically.\n"
            "Run this manually in R:\n"
            f"  {check_script}"
        )

_ensure_r_ahead_installed()

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
