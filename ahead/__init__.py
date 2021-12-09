"""Top-level package for ahead."""

__author__ = """T. Moudiki"""
__email__ = 'thierry.moudiki@gmail.com'
__version__ = '0.3.1'

from .DynamicRegressor import DynamicRegressor
from .EAT import EAT
from .Ridge2 import Ridge2Regressor
from .VAR import VAR


__all__ = ["DynamicRegressor", "EAT", "Ridge2Regressor", "VAR"]
