import numpy as np
import pandas as pd

from .calendar import load_ph_holiday_month_days
from .features import HOLIDAY_CALENDAR_PATH


def load_data(path):
    """Load the raw CSV and transform it into a clean, feature-rich DataFrame.

    This function does three things:
    1. Reads the CSV and parses dates
    2. Engineers new columns (month encoding, holidays, week-of-month)
    3. Removes invalid rows (negative values, missing data)

    The original CSV file is never modified — all changes happen in-memory only.

    Args:
        path: Path to the synthetic_lto_cdo_queue_90days.csv file.

    Returns:
        A cleaned pandas DataFrame with additional feature columns, ready
        for feature extraction by get_features().
    """
    # --- Step 1: Read raw CSV and convert date strings to datetime objects ---
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])  # "2026-01-05" → datetime(2026, 1, 5)

    # --- Step 2: Engineer new feature columns from the date ---

    # Extract raw month number (1–12) for seasonal patterns
    df["month"] = df["date"].dt.month

    # Cyclical encoding: map month 1–12 onto a circle using sin/cos
    # so December (12) is close to January (1), not far away
    month_angle = 2 * np.pi * (df["month"] - 1) / 12
    df["month_sin"] = np.sin(month_angle)
    df["month_cos"] = np.cos(month_angle)

    # Flag the last 3 days of each month — government offices see
    # a rush as deadlines approach
    df["is_end_of_month"] = (df["date"].dt.day >= (df["date"].dt.days_in_month - 2)).astype(int)

    # Load Philippine holidays and create binary flags
    holiday_md = load_ph_holiday_month_days(HOLIDAY_CALENDAR_PATH)
    if holiday_md:
        # Check if each date falls on a holiday
        df["is_holiday"] = df["date"].apply(lambda d: 1 if (d.month, d.day) in holiday_md else 0)
        # Check if TOMORROW is a holiday (pre-holiday rush: people crowd the office today)
        df["is_pre_holiday"] = df["date"].apply(
            lambda d: 1 if ((d + pd.Timedelta(days=1)).month, (d + pd.Timedelta(days=1)).day) in holiday_md else 0
        )
    else:
        # If holiday calendar file is missing, default to "no holidays"
        df["is_holiday"] = 0
        df["is_pre_holiday"] = 0

    # Calculate which week of the month this date falls in (1–5)
    # Day 1–7 → week 1, day 8–14 → week 2, etc.
    df["week_of_month"] = df["date"].dt.day.apply(lambda d: (d - 1) // 7 + 1)

    # --- Step 3: Remove invalid/corrupt rows ---
    df = df[df["waiting_time_min"] >= 0]       # Negative wait times are physically impossible
    df = df[df["queue_length_at_arrival"] >= 0] # Negative queue lengths are impossible
    df = df.dropna()                            # Drop rows with any missing values

    # --- Diagnostics: print summary so the user can verify ---
    print(f"✅ Loaded {len(df)} records")
    print(f"📊 Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # Show how many records exist per day of the week
    print("\n📅 Data distribution by day:")
    day_counts = df["day_name"].value_counts()
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]:
        if day in day_counts.index:
            print(f"   {day}: {day_counts[day]} records")

    return df
