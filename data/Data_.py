import calendar
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

np.random.seed(42)

# ==================== CONFIGURATION ====================
NUM_WEEKS = 44
START_DATE = datetime(2026, 1, 1)
DATA_DIR = Path(__file__).resolve().parent
HOLIDAY_CALENDAR_PATH = DATA_DIR / "2026-calendar-with-holidays-portrait-sunday-start-en-ph.csv"
OUTPUT_CSV_PATH = DATA_DIR / "synthetic_lto_cdo_queue_90days.csv"

# REALISTIC HOURLY WAIT TIMES (The TRUE patterns we want the model to learn)
TRUE_PATTERNS = {
    'Monday': [25, 55, 70, 60, 25, 50, 65, 55, 35],  # 8am to 4pm
    'Tuesday': [12, 28, 32, 25, 15, 20, 30, 25, 15],
    'Wednesday': [9, 19, 25, 18, 9, 15, 22, 18, 12],
    'Thursday': [11, 28, 32, 26, 16, 20, 30, 24, 15],
    'Friday': [22, 50, 65, 55, 26, 48, 60, 51, 32],
    'Saturday': [14, 28, 33, 25, 20, 24, 28, 25, 19]
}

data = []
prev_day_load = 1.0

def load_ph_holidays(calendar_path, year):
    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    holidays = set()
    if not calendar_path.exists():
        return holidays

    text = calendar_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        match = re.search(r"\b([A-Za-z]{3})\s+(\d{1,2})\s*:\s*", line)
        if not match:
            continue
        month_name = match.group(1).title()
        day = int(match.group(2))
        month = month_map.get(month_name)
        if not month:
            continue
        try:
            holidays.add(datetime(year, month, day).date())
        except ValueError:
            continue
    return holidays

holiday_dates = load_ph_holidays(HOLIDAY_CALENDAR_PATH, START_DATE.year)

print("=" * 70)
print("Generating training data with CLEAR day/hour patterns...")
print("=" * 70)

for week in range(NUM_WEEKS):
    current_date = START_DATE + timedelta(weeks=week)
    trend_factor = 1.0 + 0.01 * np.sin(2 * np.pi * week / max(1, NUM_WEEKS)) + 0.006 * (week / max(1, NUM_WEEKS))
    
    # Loop through Monday to Saturday (0-5, skip Sunday)
    for day_offset in range(6):  # Monday (0) to Saturday (5)
        day_date = current_date + timedelta(days=day_offset)
        day_name = day_date.strftime('%A')
        
        # Skip if not in patterns (shouldn't happen with range 0-5)
        if day_name not in TRUE_PATTERNS:
            continue
            
        pattern = TRUE_PATTERNS[day_name]

        month = day_date.month
        month_angle = 2 * np.pi * (month - 1) / 12
        seasonal_factor = 1.0 + 0.08 * np.sin(month_angle)
        last_day = calendar.monthrange(day_date.year, month)[1]
        is_end_of_month = 1 if day_date.day >= last_day - 2 else 0
        eom_factor = 1.06 if is_end_of_month else 1.0
        is_holiday = 1 if day_date.date() in holiday_dates else 0
        is_pre_holiday = 1 if (day_date + timedelta(days=1)).date() in holiday_dates else 0
        holiday_factor = 0.75 if is_holiday else 1.0
        pre_holiday_factor = 1.12 if is_pre_holiday else 1.0
        
        day_index = (day_date - START_DATE).days
        day_wave = 1.0 + 0.03 * np.sin(2 * np.pi * day_index / 9) + 0.02 * np.cos(2 * np.pi * day_index / 17)
        baseline_load = 1.0 + 0.04 * np.sin(2 * np.pi * day_index / 11) + 0.03 * np.cos(2 * np.pi * day_index / 23)
        day_load = 0.6 * prev_day_load + 0.4 * baseline_load
        prev_day_load = day_load
        capacity_factor = 1.03 if day_name in ['Monday', 'Friday'] else 0.98 if day_name in ['Tuesday', 'Wednesday', 'Thursday'] else 1.0

        # Generate transactions for each hour
        for hour_idx, hour in enumerate(range(8, 17)):
            hour_wave = 1.0 + 0.025 * np.sin(2 * np.pi * (hour_idx + day_index) / 9)
            base_wait = pattern[hour_idx] * seasonal_factor * eom_factor * holiday_factor * pre_holiday_factor
            base_wait = base_wait * trend_factor * day_wave * hour_wave * day_load * capacity_factor
            
            # Number of transactions this hour (more transactions = more data)
            if is_holiday:
                num_transactions = np.random.randint(4, 9)
            else:
                num_transactions = np.random.randint(8, 15)
            
            is_peak_day = 1 if day_name in ['Monday', 'Friday'] else 0
            is_peak_hour = 1 if hour in [9, 10, 11, 14, 15] else 0
            
            for t in range(num_transactions):
                minute = np.random.randint(0, 60)
                arrival_time = day_date.replace(hour=hour, minute=minute, second=0)
                
                # Add realistic variation to wait time
                wait_variation = np.random.uniform(0.85, 1.15)
                wait_time = base_wait * wait_variation
                noise = np.random.uniform(-0.08, 0.08)
                wait_time = wait_time * (1.0 + noise)
                wait_time = max(5, min(90, wait_time))
                
                # Queue length is correlated with wait time
                if wait_time > 45:
                    queue_length = np.random.randint(25, 45)
                elif wait_time > 25:
                    queue_length = np.random.randint(12, 28)
                else:
                    queue_length = np.random.randint(2, 12)

                # Nonlinear congestion: longer queues grow waits faster
                if queue_length > 18:
                    wait_time *= 1.0 + (queue_length - 18) / 90
                    wait_time = max(5, min(90, wait_time))
                
                # Service time based on congestion
                if wait_time > 45:
                    service_time = np.random.uniform(45, 75)
                elif wait_time > 25:
                    service_time = np.random.uniform(30, 50)
                else:
                    service_time = np.random.uniform(15, 35)
                
                data.append({
                    'date': day_date.date(),
                    'arrival_time': arrival_time,
                    'hour': hour,
                    'day_of_week': day_date.weekday(),
                    'day_name': day_name,
                    'month': month,
                    'month_sin': round(float(np.sin(month_angle)), 6),
                    'month_cos': round(float(np.cos(month_angle)), 6),
                    'is_end_of_month': is_end_of_month,
                    'is_holiday': is_holiday,
                    'is_pre_holiday': is_pre_holiday,
                    'is_peak_day': is_peak_day,
                    'is_peak_hour': is_peak_hour,
                    'queue_length_at_arrival': queue_length,
                    'waiting_time_min': round(wait_time, 1),
                    'service_time_min': round(service_time, 1),
                    'total_time_in_system_min': round(wait_time + service_time, 1)
                })

# Create DataFrame
df = pd.DataFrame(data)

# Create lag features (using groupby to avoid leakage)
df = df.sort_values(['date', 'arrival_time']).reset_index(drop=True)

# Add lag features by date group
df['queue_length_lag1'] = df.groupby('date')['queue_length_at_arrival'].shift(1)
df['waiting_time_lag1'] = df.groupby('date')['waiting_time_min'].shift(1)

# Fill first row of each day with realistic values
for date in df['date'].unique():
    mask = df['date'] == date
    first_idx = df[mask].index[0]
    day_name = df.loc[first_idx, 'day_name']
    
    # Use appropriate defaults based on day type
    if day_name in ['Monday', 'Friday']:
        df.loc[first_idx, 'queue_length_lag1'] = 8
        df.loc[first_idx, 'waiting_time_lag1'] = 25
    else:
        df.loc[first_idx, 'queue_length_lag1'] = 3
        df.loc[first_idx, 'waiting_time_lag1'] = 10

df = df.fillna(0)

# Add additional features
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# Save
df.to_csv(OUTPUT_CSV_PATH, index=False)

print(f"\n✅ Generated {len(df):,} transactions")
print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")

print("\n📊 VERIFICATION - Average Wait Times:")
for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']:
    day_data = df[df['day_name'] == day]
    if len(day_data) > 0:
        avg = day_data['waiting_time_min'].mean()
        print(f"   {day}: {avg:.1f} min")

print("\n⏰ HOURLY COMPARISON (Monday vs Wednesday):")
print(f"\n{'Hour':<8} {'Monday':<12} {'Wednesday':<12} {'Difference':<12}")
print("-" * 45)

for hour in range(8, 17):
    monday_avg = df[(df['day_name'] == 'Monday') & (df['hour'] == hour)]['waiting_time_min'].mean()
    wednesday_avg = df[(df['day_name'] == 'Wednesday') & (df['hour'] == hour)]['waiting_time_min'].mean()
    if not np.isnan(monday_avg) and not np.isnan(wednesday_avg):
        print(f"{hour:02d}:00   {monday_avg:>6.1f} min   {wednesday_avg:>6.1f} min   {monday_avg - wednesday_avg:>+6.1f} min")

print("\n" + "=" * 70)
print(f"✅ Data ready! Saved to {OUTPUT_CSV_PATH}")
print("✅ Next: run 'py main.py' from repo root")
print("=" * 70)