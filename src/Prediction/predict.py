"""Prediction module entry point — re-exports all prediction functions.

This file serves as the public API for the Prediction package.
Other code can import from here instead of reaching into individual submodules:

    from Prediction.predict import predict_wait_time, main

Running this file directly starts the CLI prediction interface:
    python -m Prediction.predict
"""
from pathlib import Path
import sys

# Ensure the src/ directory is on the import path
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Re-export all public functions from submodules
from Prediction.cli import display_daily_forecast, display_weekly_forecast, find_best_time, main, parse_date_input
from Prediction.inference import get_congestion_level, predict_wait_time, predict_wait_time_monte_carlo

# Explicit list of what this module exports
__all__ = [
    "display_daily_forecast",
    "display_weekly_forecast",
    "find_best_time",
    "get_congestion_level",
    "main",
    "parse_date_input",
    "predict_wait_time",
    "predict_wait_time_monte_carlo",
]


# If this file is run directly, start the CLI menu
if __name__ == "__main__":
    main()