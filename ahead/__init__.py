"""Top-level package for ahead."""
__author__ = """T. Moudiki"""
__email__ = "thierry.moudiki@gmail.com"
__version__ = "0.38.7"

import shutil
import subprocess
import sys

_R_CHECK_SCRIPT = (
    "options(repos=c(techtonique='https://r-packages.techtonique.net', "
    "CRAN='https://cloud.r-project.org')); "
    "if (!requireNamespace('ahead', quietly=TRUE)) { "
    "  tryCatch("
    "    install.packages('ahead', dependencies=TRUE), "
    "    error = function(e) message('techtonique repo failed: ', conditionMessage(e))"
    "  ); "
    "  if (!requireNamespace('ahead', quietly=TRUE)) { "
    "    if (!requireNamespace('remotes', quietly=TRUE)) install.packages('remotes'); "
    "    remotes::install_github('Techtonique/ahead'); "
    "  } "
    "  if (!requireNamespace('ahead', quietly=TRUE)) { "
    "    stop('R package ahead failed to install from all sources') "
    "  } "
    "}"
)

def _ensure_r_ahead_installed():
    if shutil.which("Rscript") is None:
        sys.exit(
            "R is required by the 'ahead' Python package but wasn't found.\n"
            "Install R from https://cran.r-project.org, then re-import ahead."
        )

    result = subprocess.run(
        ["Rscript", "-e", _R_CHECK_SCRIPT],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        sys.exit(
            "Failed to install the R 'ahead' package automatically.\n"
            f"--- R stdout ---\n{result.stdout}\n"
            f"--- R stderr ---\n{result.stderr}\n"
            "You can also try running this manually in R:\n"
            f"  {_R_CHECK_SCRIPT}"
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
    "MLARCH",
]
