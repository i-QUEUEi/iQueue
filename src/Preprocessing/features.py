from pathlib import Path

import numpy as np
import pandas as pd

from .calendar import load_ph_holiday_month_days

HOLIDAY_CALENDAR_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "2026-calendar-with-holidays-portrait-sunday-start-en-ph.csv"
)

FEATURES = [
    "hour",
    "day_of_week",
    "week_of_month",
    "month",
    "month_sin",
    "month_cos",
    "is_end_of_month",
    "is_holiday",
    "is_pre_holiday",
    "is_peak_day",
    "queue_length_at_arrival",
    "service_time_min",
    "is_weekend",
    "is_peak_hour",
    "queue_length_lag1",
    "waiting_time_lag1",
]


def get_features(df):
    X = df[FEATURES]
    y = df["waiting_time_min"]

    print("\n📊 Feature statistics:")
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
    """Convert records into a feature DataFrame matching training order."""
    if isinstance(records, dict):
        records = [records]

    calendar_path = holiday_calendar_path or HOLIDAY_CALENDAR_PATH
    holiday_md = load_ph_holiday_month_days(calendar_path)
    rows = []

    for rec in records:
        date_val = rec.get("date")
        if isinstance(date_val, str):
            date = pd.to_datetime(date_val)
        elif hasattr(date_val, "to_datetime"):
            date = pd.to_datetime(date_val)
        else:
            date = pd.Timestamp.now()

        hour = int(rec.get("hour", 9))

        dow = rec.get("day_of_week")
        if isinstance(dow, str):
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
            day_of_week_num = int(dow) if dow is not None else int(pd.Timestamp(date).weekday())

        week_of_month = (date.day - 1) // 7 + 1
        month = date.month
        angle = 2 * np.pi * (month - 1) / 12
        month_sin = float(np.sin(angle))
        month_cos = float(np.cos(angle))
        days_in_month = date.days_in_month
        is_end_of_month = 1 if date.day >= days_in_month - 2 else 0

        is_holiday = 1 if (month, date.day) in holiday_md else 0
        next_day = date + pd.Timedelta(days=1)
        is_pre_holiday = 1 if (next_day.month, next_day.day) in holiday_md else 0

        day_name = date.strftime("%A")
        is_peak_day = 1 if day_name in ["Monday", "Friday"] else 0
        is_weekend = 1 if day_name == "Saturday" else 0
        if is_peak_day:
            is_peak_hour = 1 if hour in [9, 10, 11, 13, 14, 15] else 0
        else:
            is_peak_hour = 1 if hour in [9, 10, 14, 15] else 0

        queue = float(rec.get("queue_length_at_arrival", rec.get("queue_length", 5)))
        queue_length_lag1 = max(1.0, queue - 3.0)
        waiting_time_lag1 = max(1.0, queue * 1.5)
        service_time_min = float(rec.get("service_time_min", 35.0))

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

    return pd.DataFrame(rows, columns=FEATURES)
