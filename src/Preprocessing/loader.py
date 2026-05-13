import numpy as np
import pandas as pd

from .calendar import load_ph_holiday_month_days
from .features import HOLIDAY_CALENDAR_PATH


def load_data(path):
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])

    df["month"] = df["date"].dt.month
    month_angle = 2 * np.pi * (df["month"] - 1) / 12
    df["month_sin"] = np.sin(month_angle)
    df["month_cos"] = np.cos(month_angle)
    df["is_end_of_month"] = (df["date"].dt.day >= (df["date"].dt.days_in_month - 2)).astype(int)

    holiday_md = load_ph_holiday_month_days(HOLIDAY_CALENDAR_PATH)
    if holiday_md:
        df["is_holiday"] = df["date"].apply(lambda d: 1 if (d.month, d.day) in holiday_md else 0)
        df["is_pre_holiday"] = df["date"].apply(
            lambda d: 1 if ((d + pd.Timedelta(days=1)).month, (d + pd.Timedelta(days=1)).day) in holiday_md else 0
        )
    else:
        df["is_holiday"] = 0
        df["is_pre_holiday"] = 0

    df["week_of_month"] = df["date"].dt.day.apply(lambda d: (d - 1) // 7 + 1)

    df = df[df["waiting_time_min"] >= 0]
    df = df[df["queue_length_at_arrival"] >= 0]
    df = df.dropna()

    print(f"✅ Loaded {len(df)} records")
    print(f"📊 Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    print("\n📅 Data distribution by day:")
    day_counts = df["day_name"].value_counts()
    for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]:
        if day in day_counts.index:
            print(f"   {day}: {day_counts[day]} records")

    return df
