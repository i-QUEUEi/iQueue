"""Preprocessing convenience hub — re-exports all preprocessing functions.

This module doesn't contain any logic. It imports functions from the other
3 preprocessing files (loader, features, calendar) and re-exports them,
so the rest of the codebase can import everything from one place:

    from Preprocessing.preprocess import load_data, get_features

Instead of:

    from Preprocessing.loader import load_data
    from Preprocessing.features import get_features
    from Preprocessing.calendar import load_ph_holiday_month_days
"""

from .calendar import load_ph_holiday_month_days
from .features import FEATURES, build_feature_dataframe, get_features
from .loader import load_data

# Explicit list of what this module exports when someone does `from preprocess import *`
__all__ = [
    "FEATURES",
    "build_feature_dataframe",
    "get_features",
    "load_data",
    "load_ph_holiday_month_days",
]