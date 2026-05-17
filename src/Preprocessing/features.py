from pathlib import Path

import numpy as np
import pandas as pd

from .calendar import load_ph_holiday_month_days

# Path to the Philippine holiday calendar file, located in the data/ directory
HOLIDAY_CALENDAR_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "2026-calendar-with-holidays-portrait-sunday-start-en-ph.csv"
)

# ==================== FEATURE LIST ====================
# These are the exact 16 columns the model is allowed to see during training.
# The target column (waiting_time_min) is deliberately EXCLUDED — that's what
# the model is trying to predict, so it can't be an input.
#
# Order matters: the model expects features in this exact sequence.
FEATURES = [
    "hour",                    # What time of day (8–16)
    "day_of_week",             # Monday=0 through Saturday=5
    "week_of_month",           # Week 1, 2, 3, 4, or 5 within the month
    "month",                   # January=1 through December=12
    "month_sin",               # Cyclical month encoding (sine component)
    "month_cos",               # Cyclical month encoding (cosine component)
    "is_end_of_month",         # 1 if within last 3 days of month, else 0
    "is_holiday",              # 1 if this is a Philippine holiday, else 0
    "is_pre_holiday",          # 1 if tomorrow is a holiday, else 0
    "is_peak_day",             # 1 if Monday or Friday (busiest days), else 0
    "queue_length_at_arrival", # How many people are already in line when you arrive
    "service_time_min",        # How long the current transaction takes (minutes)
    "is_weekend",              # 1 if Saturday, else 0
    "is_peak_hour",            # 1 if during peak hours (9-11am, 2-3pm), else 0
    "queue_length_lag1",       # Queue length of the PREVIOUS transaction (memory)
    "waiting_time_lag1",       # Wait time of the PREVIOUS transaction (memory)
]


def get_features(df):
    """Extract the 16 input features (X) and target variable (y) from the DataFrame.

    This is the bridge between raw data and model training — it selects
    exactly which columns the model can see.

    Args:
        df: Cleaned DataFrame from load_data().

    Returns:
        X: DataFrame with 16 feature columns.
        y: Series with waiting_time_min (what we're predicting).
        FEATURES: The list of feature names (for reference).
    """
    X = df[FEATURES]              # Select only the 16 allowed input columns
    y = df["waiting_time_min"]    # Extract the target: what we're predicting

    # Print feature statistics so the user can verify the data distribution
    print("\n[INFO] Feature statistics:")
    print(f"   Peak days (Mon/Fri): {X['is_peak_day'].sum()} records")
    print(f"   Non-peak days: {len(X) - X['is_peak_day'].sum()} records")
    print(f"   Peak hours: {X['is_peak_hour'].sum()} records")
    print(f"   Week-of-month range: {X['week_of_month'].min()} to {X['week_of_month'].max()}")
    print(f"   Month range: {X['month'].min()} to {X['month'].max()}")
    print(f"   End-of-month rows: {X['is_end_of_month'].sum()} records")
    print(f"   Holiday rows: {X['is_holiday'].sum()} records")
    print(f"   Pre-holiday rows: {X['is_pre_holiday'].sum()} records")

    return X, y, FEATURES


def build_feature_dataframe(records, holiday_calendar_path=None):
    """Convert raw prediction records into a feature DataFrame matching training format.

    This is used during PREDICTION (not training) — when a user asks
    "what's the wait at 9am on Monday?", this function builds the exact
    same 16-column DataFrame that the model was trained on.

    Args:
        records: A single dict or list of dicts with fields like
                 {"date": "2026-03-15", "hour": 9, "day_of_week": 0, ...}.
        holiday_calendar_path: Optional override for the holiday calendar path.

    Returns:
        A pandas DataFrame with columns in the exact FEATURES order,
        ready to be passed to model.predict().
    """
    # Accept either a single record or a list of records
    if isinstance(records, dict):
        records = [records]

    # Load the holiday calendar for holiday flag computation
    calendar_path = holiday_calendar_path or HOLIDAY_CALENDAR_PATH
    holiday_md = load_ph_holiday_month_days(calendar_path)
    rows = []

    # Process each record into a complete feature row
    for rec in records:
        # --- Parse the date ---
        date_val = rec.get("date")
        if isinstance(date_val, str):
            date = pd.to_datetime(date_val)       # String → datetime
        elif hasattr(date_val, "to_datetime"):
            date = pd.to_datetime(date_val)        # Pandas-compatible → datetime
        else:
            date = pd.Timestamp.now()              # Fallback: use current time

        # Extract hour, defaulting to 9am if not provided
        hour = int(rec.get("hour", 9))

        # --- Parse day_of_week (handles both string and integer) ---
        dow = rec.get("day_of_week")
        if isinstance(dow, str):
            # Convert day name to number: "Monday" → 0, "Saturday" → 5
            day_map = {
                "Monday": 0,
                "Tuesday": 1,
                "Wednesday": 2,
                "Thursday": 3,
                "Friday": 4,
                "Saturday": 5,
            }
            day_of_week_num = day_map.get(dow, pd.Timestamp(date).weekday())
        else:
            # Already a number, or derive from the date
            day_of_week_num = int(dow) if dow is not None else int(pd.Timestamp(date).weekday())

        # --- Compute derived features (same logic as loader.py) ---
        week_of_month = (date.day - 1) // 7 + 1       # Which week of the month (1-5)
        month = date.month                              # Month number (1-12)
        angle = 2 * np.pi * (month - 1) / 12           # Angle for cyclical encoding
        month_sin = float(np.sin(angle))                # Sine component
        month_cos = float(np.cos(angle))                # Cosine component
        days_in_month = date.days_in_month              # Total days in this month
        is_end_of_month = 1 if date.day >= days_in_month - 2 else 0  # Last 3 days flag

        # --- Holiday flags ---
        is_holiday = 1 if (month, date.day) in holiday_md else 0
        next_day = date + pd.Timedelta(days=1)
        is_pre_holiday = 1 if (next_day.month, next_day.day) in holiday_md else 0

        # --- Day-type flags ---
        day_name = date.strftime("%A")                  # "Monday", "Tuesday", etc.
        is_peak_day = 1 if day_name in ["Monday", "Friday"] else 0
        is_weekend = 1 if day_name == "Saturday" else 0

        # Peak hours differ between peak days and non-peak days
        if is_peak_day:
            is_peak_hour = 1 if hour in [9, 10, 11, 13, 14, 15] else 0
        else:
            is_peak_hour = 1 if hour in [9, 10, 14, 15] else 0

        # --- Queue and lag features (estimated from input or defaults) ---
        queue = float(rec.get("queue_length_at_arrival", rec.get("queue_length", 5)))
        queue_length_lag1 = max(1.0, queue - 3.0)       # Estimate: previous queue was ~3 less
        waiting_time_lag1 = max(1.0, queue * 1.5)        # Estimate: wait ≈ 1.5× queue length
        service_time_min = float(rec.get("service_time_min", 35.0))  # Default: 35 min

        # --- Assemble the complete feature row ---
        rows.append(
            {
                "hour": hour,
                "day_of_week": day_of_week_num,
                "week_of_month": week_of_month,
                "month": month,
                "month_sin": month_sin,
                "month_cos": month_cos,
                "is_end_of_month": is_end_of_month,
                "is_holiday": is_holiday,
                "is_pre_holiday": is_pre_holiday,
                "is_peak_day": is_peak_day,
                "queue_length_at_arrival": queue,
                "service_time_min": service_time_min,
                "is_weekend": is_weekend,
                "is_peak_hour": is_peak_hour,
                "queue_length_lag1": queue_length_lag1,
                "waiting_time_lag1": waiting_time_lag1,
            }
        )

    # Return DataFrame with columns in the exact FEATURES order (must match training)
    return pd.DataFrame(rows, columns=FEATURES)
