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


def build_feature_dataframe(records):
    """Convert input record or list of records (dicts) into a feature DataFrame matching training order.

    Expected fields per record: `date` (YYYY-MM-DD) or `date` as datetime, `hour`, `day_of_week` (string or int),
    `queue_length_at_arrival` (number). Missing auxiliary features are filled with sensible defaults.
    """
    if isinstance(records, dict):
        records = [records]

    rows = []
    # simple holiday loader uses same calendar parsing as load_ph_holiday_month_days
    holiday_md = load_ph_holiday_month_days(HOLIDAY_CALENDAR_PATH)

    for rec in records:
        # date handling
        date_val = rec.get('date')
        if isinstance(date_val, str):
            date = pd.to_datetime(date_val)
        elif hasattr(date_val, 'to_datetime'):
            date = pd.to_datetime(date_val)
        else:
            date = pd.Timestamp.now()

        hour = int(rec.get('hour', 9))

        # day_of_week can be string or numeric
        dow = rec.get('day_of_week')
        if isinstance(dow, str):
            day_of_week = pd.Timestamp(date).day_name()
            # map string to numeric
            day_map = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,'Friday':4,'Saturday':5}
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

        # peak day/weekend
        day_name = date.strftime('%A')
        is_peak_day = 1 if day_name in ['Monday', 'Friday'] else 0
        is_weekend = 1 if day_name == 'Saturday' else 0
        if is_peak_day:
            is_peak_hour = 1 if hour in [9,10,11,13,14,15] else 0
        else:
            is_peak_hour = 1 if hour in [9,10,14,15] else 0

        queue = float(rec.get('queue_length_at_arrival', rec.get('queue_length', 5)))
        # reasonable defaults for lags and service time
        queue_length_lag1 = max(1.0, queue - 3.0)
        waiting_time_lag1 = max(1.0, queue * 1.5)
        service_time_min = float(rec.get('service_time_min', 35.0))

        row = {
            'hour': hour,
            'day_of_week': day_of_week_num,
            'week_of_month': week_of_month,
            'month': month,
            'month_sin': month_sin,
            'month_cos': month_cos,
            'is_end_of_month': is_end_of_month,
            'is_holiday': is_holiday,
            'is_pre_holiday': is_pre_holiday,
            'is_peak_day': is_peak_day,
            'queue_length_at_arrival': queue,
            'service_time_min': service_time_min,
            'is_weekend': is_weekend,
            'is_peak_hour': is_peak_hour,
            'queue_length_lag1': queue_length_lag1,
            'waiting_time_lag1': waiting_time_lag1
        }
        rows.append(row)

    feat_order = [
        'hour', 'day_of_week', 'week_of_month', 'month', 'month_sin', 'month_cos',
        'is_end_of_month', 'is_holiday', 'is_pre_holiday', 'is_peak_day',
        'queue_length_at_arrival', 'service_time_min', 'is_weekend', 'is_peak_hour',
        'queue_length_lag1', 'waiting_time_lag1'
    ]

    X = pd.DataFrame(rows, columns=feat_order)
    return X