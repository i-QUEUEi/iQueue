import re
from pathlib import Path

import numpy as np
import pandas as pd

HOLIDAY_CALENDAR_PATH = Path(__file__).resolve().parents[1] / "data" / "2026-calendar-with-holidays-portrait-sunday-start-en-ph.csv"

def load_ph_holiday_month_days(calendar_path):
    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    holiday_md = set()
    if not calendar_path.exists():
        return holiday_md

    text = calendar_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        match = re.search(r"\b([A-Za-z]{3})\s+(\d{1,2})\s*:\s*", line)
        if not match:
            continue
        month_name = match.group(1).title()
        day = int(match.group(2))
        month = month_map.get(month_name)
        if month:
            holiday_md.add((month, day))
    return holiday_md

def load_data(path):
    df = pd.read_csv(path)
    
    # Ensure date is properly parsed
    df['date'] = pd.to_datetime(df['date'])

    df['month'] = df['date'].dt.month
    month_angle = 2 * np.pi * (df['month'] - 1) / 12
    df['month_sin'] = np.sin(month_angle)
    df['month_cos'] = np.cos(month_angle)
    df['is_end_of_month'] = (df['date'].dt.day >= (df['date'].dt.days_in_month - 2)).astype(int)

    holiday_md = load_ph_holiday_month_days(HOLIDAY_CALENDAR_PATH)
    if holiday_md:
        df['is_holiday'] = df['date'].apply(lambda d: 1 if (d.month, d.day) in holiday_md else 0)
        df['is_pre_holiday'] = df['date'].apply(
            lambda d: 1 if ((d + pd.Timedelta(days=1)).month, (d + pd.Timedelta(days=1)).day) in holiday_md else 0
        )
    else:
        df['is_holiday'] = 0
        df['is_pre_holiday'] = 0
    
    # Derive week_of_month (1-4) so Week-1 Monday differs from Week-4 Monday
    df['week_of_month'] = df['date'].dt.day.apply(lambda d: (d - 1) // 7 + 1)
    
    # Remove any invalid rows
    df = df[df['waiting_time_min'] >= 0]
    df = df[df['queue_length_at_arrival'] >= 0]
    
    df = df.dropna()
    
    print(f"✅ Loaded {len(df)} records")
    print(f"📊 Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    print("\n📅 Data distribution by day:")
    day_counts = df['day_name'].value_counts()
    for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']:
        if day in day_counts.index:
            print(f"   {day}: {day_counts[day]} records")
    
    return df

def get_features(df):
    # Features in the correct order for the model
    features = [
        'hour',
        'day_of_week',
        'week_of_month',
        'month',
        'month_sin',
        'month_cos',
        'is_end_of_month',
        'is_holiday',
        'is_pre_holiday',
        'is_peak_day',
        'queue_length_at_arrival',
        'service_time_min',
        'is_weekend',
        'is_peak_hour',
        'queue_length_lag1',
        'waiting_time_lag1'
    ]
    
    X = df[features]
    y = df['waiting_time_min']
    
    print(f"\n📊 Feature statistics:")
    print(f"   Peak days (Mon/Fri): {X['is_peak_day'].sum()} records")
    print(f"   Non-peak days: {len(X) - X['is_peak_day'].sum()} records")
    print(f"   Peak hours: {X['is_peak_hour'].sum()} records")
    print(f"   Week-of-month range: {X['week_of_month'].min()} to {X['week_of_month'].max()}")
    print(f"   Month range: {X['month'].min()} to {X['month'].max()}")
    print(f"   End-of-month rows: {X['is_end_of_month'].sum()} records")
    print(f"   Holiday rows: {X['is_holiday'].sum()} records")
    print(f"   Pre-holiday rows: {X['is_pre_holiday'].sum()} records")
    
    return X, y, features