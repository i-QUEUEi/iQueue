"""Prediction context — loads the trained model and builds historical pattern lookup tables.

This module runs at IMPORT TIME (when predict.py starts). It:
1. Loads the saved .pkl model from disk
2. Reads the training CSV to compute historical averages
3. Builds pattern lookup tables organized by month → week → day → hour
4. Loads the holiday calendar

Everything computed here is stored as module-level variables that other
modules (inference.py, cli.py) import and use directly.
"""
from pathlib import Path
import sys

import joblib
import pandas as pd

# Add the src/ directory to Python's import path so we can import Preprocessing
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Preprocessing.calendar import load_ph_holiday_month_days
from .patterns import build_pattern_maps, get_pattern_value

# ==================== FILE PATHS ====================
BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "queue_model.pkl"           # Trained model file
DATA_PATH = BASE_DIR / "data" / "synthetic_lto_cdo_queue_90days.csv"  # Training data
HOLIDAY_CALENDAR_PATH = (
    BASE_DIR / "data" / "2026-calendar-with-holidays-portrait-sunday-start-en-ph.csv"
)

# ==================== LOAD THE TRAINED MODEL ====================
# joblib.load() deserializes the .pkl file back into a fully trained model object
# (including all 500 trees for Random Forest, or 250 stages for Gradient Boosting)
model = joblib.load(MODEL_PATH)

# ==================== BUILD PATTERN LOOKUP TABLES ====================
# Read the training CSV and compute historical averages for every
# day/month/week/hour combination. These are used to fill in features
# like queue_length when making predictions.
print("📊 Calculating actual queue patterns from training data...")

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df["week_of_month"] = df["date"].dt.day.apply(lambda d: (d - 1) // 7 + 1)
df["month"] = df["date"].dt.month

# Build lookup tables for queue length and wait time
# These are 4-level nested dicts: day → month → week → hour → average value
queue_maps = build_pattern_maps(df, "queue_length_at_arrival")
wait_maps = build_pattern_maps(df, "waiting_time_min")

# Print sample values to verify the patterns were computed correctly
print("\n✅ Loaded date-aware queue patterns (12 months × 4 weeks × 6 days × 9 hours):")
print(
    "   Monday Wk1 9am avg queue: {} people".format(
        round(get_pattern_value(queue_maps, "Monday", 1, 1, 9, 5), 1)
    )
)
print(
    "   Monday Wk4 9am avg queue: {} people".format(
        round(get_pattern_value(queue_maps, "Monday", 1, 4, 9, 5), 1)
    )
)
print(
    "   Wednesday Wk1 9am avg queue: {} people".format(
        round(get_pattern_value(queue_maps, "Wednesday", 1, 1, 9, 5), 1)
    )
)
print(
    "   Wednesday Wk4 9am avg queue: {} people".format(
        round(get_pattern_value(queue_maps, "Wednesday", 1, 4, 9, 5), 1)
    )
)

# ==================== LOAD HOLIDAY CALENDAR ====================
# Set of (month, day) tuples for quick "is this a holiday?" lookups
holiday_month_days = load_ph_holiday_month_days(HOLIDAY_CALENDAR_PATH)

# ==================== COMPUTE GLOBAL AVERAGES ====================
# Average service time across all transactions (used as default in predictions)
avg_service_time = df["service_time_min"].mean()
