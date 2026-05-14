"""Pattern map builder — creates multi-level lookup tables from historical data.

This module builds nested dictionaries that store the AVERAGE value of a column
(like queue_length or waiting_time) at different levels of detail:
- Level 1: day + hour (broadest)
- Level 2: day + week_of_month + hour
- Level 3: day + month + hour
- Level 4: day + month + week_of_month + hour (most specific)

When making predictions, we try the most specific level first and
fall back to broader levels if no data exists for that combination.
"""
import pandas as pd

# Days the LTO office is open (Sunday is closed)
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]


def build_pattern_maps(df, value_col):
    """Build 4-level nested lookup tables from historical data.

    For each day of the week, computes the average value of `value_col`
    at increasingly specific granularities.

    Args:
        df: DataFrame with columns 'day_name', 'hour', 'week_of_month', 'month',
            and whatever `value_col` is (e.g., 'queue_length_at_arrival').
        value_col: The column to compute averages for.

    Returns:
        Dictionary with 4 lookup tables:
        - day_hour: avg by day + hour (broadest)
        - day_week_hour: avg by day + week + hour
        - day_month_hour: avg by day + month + hour
        - day_month_week_hour: avg by day + month + week + hour (most specific)
    """
    # Initialize the 4 lookup tables
    day_hour = {}               # Level 1: day → hour → avg
    day_week_hour = {}          # Level 2: day → week → hour → avg
    day_month_hour = {}         # Level 3: day → month → hour → avg
    day_month_week_hour = {}    # Level 4: day → month → week → hour → avg

    # Build lookup tables for each day of the week
    for day in DAYS:
        day_df = df[df["day_name"] == day]  # Filter to just this day
        day_hour[day] = {}
        day_week_hour[day] = {}
        day_month_hour[day] = {}
        day_month_week_hour[day] = {}

        # Level 1: Average by day + hour only (e.g., "Monday 9am average across ALL weeks")
        for hour in range(8, 17):
            day_hour[day][hour] = day_df[day_df["hour"] == hour][value_col].mean()

        # Level 2: Average by day + week_of_month + hour
        # (e.g., "Monday week 1 at 9am" vs "Monday week 4 at 9am")
        for week in range(1, 6):
            week_df = day_df[day_df["week_of_month"] == week]
            day_week_hour[day][week] = {}
            for hour in range(8, 17):
                day_week_hour[day][week][hour] = week_df[week_df["hour"] == hour][value_col].mean()

        # Level 3 & 4: Add month granularity
        for month in range(1, 13):
            month_df = day_df[day_df["month"] == month]

            # Level 3: Average by day + month + hour
            # (e.g., "Mondays in January at 9am")
            day_month_hour[day][month] = {}
            for hour in range(8, 17):
                day_month_hour[day][month][hour] = month_df[month_df["hour"] == hour][value_col].mean()

            # Level 4: Average by day + month + week + hour (most specific)
            # (e.g., "Monday, week 2 of January, at 9am")
            day_month_week_hour[day][month] = {}
            for week in range(1, 6):
                week_df = month_df[month_df["week_of_month"] == week]
                day_month_week_hour[day][month][week] = {}
                for hour in range(8, 17):
                    day_month_week_hour[day][month][week][hour] = week_df[week_df["hour"] == hour][value_col].mean()

    return {
        "day_hour": day_hour,
        "day_week_hour": day_week_hour,
        "day_month_hour": day_month_hour,
        "day_month_week_hour": day_month_week_hour,
    }


def get_pattern_value(pattern_maps, day_name, month, week, hour, default_value):
    """Look up the average value, falling back from most specific to broadest.

    Tries 4 levels in order (most specific first):
    1. day + month + week + hour  →  "Monday, Jan, week 2, 9am"
    2. day + month + hour         →  "Monday, Jan, 9am"
    3. day + week + hour          →  "Monday, week 2, 9am"
    4. day + hour                 →  "Monday, 9am"

    If ALL levels return NaN (no data), falls back to the default_value.

    Args:
        pattern_maps: The lookup tables from build_pattern_maps().
        day_name: Day of the week (e.g., "Monday").
        month: Month number (1–12).
        week: Week of the month (1–5).
        hour: Hour of the day (8–16).
        default_value: Fallback value if no historical data exists.

    Returns:
        The average value from the most specific available level, or default_value.
    """
    # Try Level 4 (most specific): day + month + week + hour
    value = pattern_maps["day_month_week_hour"][day_name][month][week][hour]
    if not pd.isna(value):
        return float(value)

    # Try Level 3: day + month + hour
    value = pattern_maps["day_month_hour"][day_name][month][hour]
    if not pd.isna(value):
        return float(value)

    # Try Level 2: day + week + hour
    value = pattern_maps["day_week_hour"][day_name][week][hour]
    if not pd.isna(value):
        return float(value)

    # Try Level 1 (broadest): day + hour
    value = pattern_maps["day_hour"][day_name][hour]
    if not pd.isna(value):
        return float(value)

    # No data at any level — use the provided default
    return float(default_value)
