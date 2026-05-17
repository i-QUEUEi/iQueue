"""Data_.py — Synthetic data generator for the iQueue training pipeline.

This script generates ~28,000 realistic LTO CDO queue transactions spanning
44 weeks (roughly 10 months). The data is NOT random noise — it uses 17+
mathematical modifiers to simulate real-world queue patterns:
- Monday and Friday are busier than mid-week
- 9-11am are peak hours
- Holidays reduce traffic; pre-holidays increase it
- End-of-month sees a rush
- Queue length and wait time are correlated

The output CSV is the ONLY file written by the preprocessing pipeline.
All other preprocessing happens in-memory.

Usage: python data/Data_.py
Output: data/synthetic_lto_cdo_queue_90days.csv
"""
import calendar
import re
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

# Lock the random seed for reproducibility — running this script twice
# produces the EXACT same data, so model results are comparable
np.random.seed(42)

# ==================== CONFIGURATION ====================
NUM_WEEKS = 44                                  # How many weeks of data to generate (~10 months)
START_DATE = datetime(2026, 1, 1)               # First day of generated data
DATA_DIR = Path(__file__).resolve().parent       # data/ directory
HOLIDAY_CALENDAR_PATH = DATA_DIR / "2026-calendar-with-holidays-portrait-sunday-start-en-ph.csv"
OUTPUT_CSV_PATH = DATA_DIR / "synthetic_lto_cdo_queue_90days.csv"

# ==================== TRUE PATTERNS (GROUND TRUTH) ====================
# These are the BASE wait times (in minutes) for each hour of each day.
# They represent the "ideal" values the model should learn.
# Each list has 9 values: [8am, 9am, 10am, 11am, 12pm, 1pm, 2pm, 3pm, 4pm]
TRUE_PATTERNS = {
    'Monday': [25, 55, 70, 60, 25, 50, 65, 55, 35],      # Busiest day: peaks at 10am (70 min)
    'Tuesday': [12, 28, 32, 25, 15, 20, 30, 25, 15],      # Moderate day
    'Wednesday': [9, 19, 25, 18, 9, 15, 22, 18, 12],      # Calmest weekday
    'Thursday': [11, 28, 32, 26, 16, 20, 30, 24, 15],     # Similar to Tuesday
    'Friday': [22, 50, 65, 55, 26, 48, 60, 51, 32],       # Second busiest (end-of-week rush)
    'Saturday': [14, 28, 33, 25, 20, 24, 28, 25, 19]      # Half-day, moderate
}

# Accumulator for all generated transaction records
data = []
# Tracks the previous day's "load" for autocorrelation (busy days tend to follow busy days)
prev_day_load = 1.0


def load_ph_holidays(calendar_path, year):
    """Parse Philippine holiday dates from the calendar CSV file.

    Reads a text file where holidays are listed as lines like:
        "Jan 1 : New Year's Day"
    and converts them to a set of datetime.date objects.

    Args:
        calendar_path: Path to the holiday calendar file.
        year: The year to create dates for.

    Returns:
        Set of datetime.date objects for each holiday.
    """
    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    holidays = set()
    if not calendar_path.exists():
        return holidays  # No calendar file → no holidays

    text = calendar_path.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        # Regex: find "Jan 1 :" pattern
        match = re.search(r"\b([A-Za-z]{3})\s+(\d{1,2})\s*:\s*", line)
        if not match:
            continue
        month_name = match.group(1).title()  # Normalize: "jan" → "Jan"
        day = int(match.group(2))            # "1" → 1
        month = month_map.get(month_name)
        if not month:
            continue
        try:
            holidays.add(datetime(year, month, day).date())
        except ValueError:
            continue  # Skip invalid dates (e.g., Feb 30)
    return holidays


# Load holidays before the main generation loop
holiday_dates = load_ph_holidays(HOLIDAY_CALENDAR_PATH, START_DATE.year)

print("=" * 70)
print("Generating training data with CLEAR day/hour patterns...")
print("=" * 70)

# ==================== MAIN DATA GENERATION LOOP ====================
# Outer loop: iterate through each week
for week in range(NUM_WEEKS):
    # Calculate the Monday of this week (snap to week start regardless of START_DATE's day)
    # Without this, weeks starting mid-week (e.g. Thursday) would skip Wednesday entirely
    raw_date = START_DATE + timedelta(weeks=week)
    current_date = raw_date - timedelta(days=raw_date.weekday())  # weekday() 0=Mon, so this snaps to Monday


    # FACTOR: Long-term trend — slight sinusoidal drift + linear growth over time
    # Makes the data look like queue volumes slowly evolve (not perfectly static)
    trend_factor = 1.0 + 0.01 * np.sin(2 * np.pi * week / max(1, NUM_WEEKS)) + 0.006 * (week / max(1, NUM_WEEKS))

    # Inner loop: iterate through Monday (0) to Saturday (5)
    for day_offset in range(6):
        day_date = current_date + timedelta(days=day_offset)
        day_name = day_date.strftime('%A')  # "Monday", "Tuesday", etc.

        # Safety check: skip if day name not in our patterns
        if day_name not in TRUE_PATTERNS:
            continue

        # Get the base wait-time profile for this day of the week
        pattern = TRUE_PATTERNS[day_name]

        # ---------- COMPUTE DAILY MODIFIERS ----------

        # FACTOR: Seasonal variation — sine wave over the year
        # Summer months are slightly busier than winter months
        month = day_date.month
        month_angle = 2 * np.pi * (month - 1) / 12
        seasonal_factor = 1.0 + 0.08 * np.sin(month_angle)

        # FACTOR: End-of-month rush — last 3 days of each month
        # People rush to complete license renewals before month-end deadlines
        last_day = calendar.monthrange(day_date.year, month)[1]
        is_end_of_month = 1 if day_date.day >= last_day - 2 else 0
        eom_factor = 1.06 if is_end_of_month else 1.0  # 6% increase

        # FACTOR: Holiday effect — queues are shorter ON holidays (fewer people come)
        is_holiday = 1 if day_date.date() in holiday_dates else 0
        is_pre_holiday = 1 if (day_date + timedelta(days=1)).date() in holiday_dates else 0
        holiday_factor = 0.75 if is_holiday else 1.0       # 25% reduction on holidays
        pre_holiday_factor = 1.12 if is_pre_holiday else 1.0  # 12% increase before holidays

        # FACTOR: Multi-frequency day wave — adds irregular daily variation
        # Two sine waves with different periods (9 and 17 days) create non-repeating patterns
        day_index = (day_date - START_DATE).days
        day_wave = 1.0 + 0.03 * np.sin(2 * np.pi * day_index / 9) + 0.02 * np.cos(2 * np.pi * day_index / 17)

        # FACTOR: Autocorrelated daily load — busy days tend to follow busy days
        # Weighted average: 60% from yesterday's load + 40% from a baseline wave
        baseline_load = 1.0 + 0.04 * np.sin(2 * np.pi * day_index / 11) + 0.03 * np.cos(2 * np.pi * day_index / 23)
        day_load = 0.6 * prev_day_load + 0.4 * baseline_load
        prev_day_load = day_load  # Remember today's load for tomorrow

        # FACTOR: Day-of-week capacity — Mon/Fri slightly amplified, mid-week dampened
        capacity_factor = 1.03 if day_name in ['Monday', 'Friday'] else 0.98 if day_name in ['Tuesday', 'Wednesday', 'Thursday'] else 1.0

        # ---------- HOURLY TRANSACTION GENERATION ----------
        # Loop through each working hour (8am to 4pm)
        for hour_idx, hour in enumerate(range(8, 17)):
            # FACTOR: Hourly micro-wave — slight variation within each hour
            hour_wave = 1.0 + 0.025 * np.sin(2 * np.pi * (hour_idx + day_index) / 9)

            # Combine ALL daily + hourly factors with the base pattern
            # Final wait = pattern × seasonal × end_of_month × holiday × trend × waves × load × capacity
            base_wait = pattern[hour_idx] * seasonal_factor * eom_factor * holiday_factor * pre_holiday_factor
            base_wait = base_wait * trend_factor * day_wave * hour_wave * day_load * capacity_factor

            # FACTOR: Transaction count variation — holidays have fewer transactions
            if is_holiday:
                num_transactions = np.random.randint(4, 9)   # Fewer people show up
            else:
                num_transactions = np.random.randint(8, 15)  # Normal volume

            # Set binary peak flags
            is_peak_day = 1 if day_name in ['Monday', 'Friday'] else 0
            is_peak_hour = 1 if hour in [9, 10, 11, 14, 15] else 0

            # Generate individual transactions for this hour
            for t in range(num_transactions):
                # Random arrival minute within the hour
                minute = np.random.randint(0, 60)
                arrival_time = day_date.replace(hour=hour, minute=minute, second=0)

                # FACTOR: Per-transaction noise — ±15% multiplicative + ±8% additive
                # Makes each transaction slightly different (no two are identical)
                wait_variation = np.random.uniform(0.85, 1.15)
                wait_time = base_wait * wait_variation
                noise = np.random.uniform(-0.08, 0.08)
                wait_time = wait_time * (1.0 + noise)
                wait_time = max(5, min(90, wait_time))  # Clamp to realistic range [5, 90] min

                # FACTOR: Queue length correlated with wait time
                # High wait → many people in queue; low wait → few people
                if wait_time > 45:
                    queue_length = np.random.randint(25, 45)   # HIGH congestion
                elif wait_time > 25:
                    queue_length = np.random.randint(12, 28)   # MODERATE congestion
                else:
                    queue_length = np.random.randint(2, 12)    # LOW congestion

                # FACTOR: Nonlinear congestion amplification
                # When queue exceeds 18, each additional person makes wait grow faster
                if queue_length > 18:
                    wait_time *= 1.0 + (queue_length - 18) / 90
                    wait_time = max(5, min(90, wait_time))

                # FACTOR: Service time correlated with congestion
                # Busier hours → more complex transactions → longer service
                if wait_time > 45:
                    service_time = np.random.uniform(45, 75)   # Complex transactions
                elif wait_time > 25:
                    service_time = np.random.uniform(30, 50)   # Medium transactions
                else:
                    service_time = np.random.uniform(15, 35)   # Quick transactions

                # Append this transaction as a complete record
                data.append({
                    'date': day_date.date(),
                    'arrival_time': arrival_time,
                    'hour': hour,
                    'day_of_week': day_date.weekday(),          # Monday=0, Saturday=5
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

# ==================== POST-PROCESSING ====================

# Convert the list of dicts into a DataFrame
df = pd.DataFrame(data)

# Sort by date and arrival time (chronological order within each day)
df = df.sort_values(['date', 'arrival_time']).reset_index(drop=True)

# ===== CREATE LAG FEATURES =====
# Lag = "what happened in the PREVIOUS transaction?"
# The model uses this as "memory" — if the previous person waited 50 min,
# the current person is likely in for a long wait too.
# Using groupby('date') prevents leaking data across days (yesterday's last
# transaction shouldn't be the lag for today's first transaction).
df['queue_length_lag1'] = df.groupby('date')['queue_length_at_arrival'].shift(1)
df['waiting_time_lag1'] = df.groupby('date')['waiting_time_min'].shift(1)

# Fill the first transaction of each day (which has no "previous" transaction)
# with realistic default values based on day type
for date in df['date'].unique():
    mask = df['date'] == date
    first_idx = df[mask].index[0]       # Get the index of the first transaction
    day_name = df.loc[first_idx, 'day_name']

    # Peak days (Mon/Fri) start with higher defaults than mid-week days
    if day_name in ['Monday', 'Friday']:
        df.loc[first_idx, 'queue_length_lag1'] = 8     # Expect ~8 people at 8am
        df.loc[first_idx, 'waiting_time_lag1'] = 25    # Expect ~25 min wait at 8am
    else:
        df.loc[first_idx, 'queue_length_lag1'] = 3     # Calm morning
        df.loc[first_idx, 'waiting_time_lag1'] = 10    # Short initial wait

# Fill any remaining NaN values with 0
df = df.fillna(0)

# Add weekend flag (Saturday = 1, weekdays = 0)
df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

# ==================== SAVE THE CSV ====================
# This is the ONLY file write in the entire preprocessing pipeline.
# The original CSV is never modified after this point.
df.to_csv(OUTPUT_CSV_PATH, index=False)

# ==================== VERIFICATION OUTPUT ====================
# Print statistics so the user can verify the data looks realistic
print(f"\n✅ Generated {len(df):,} transactions")
print(f"📅 Date range: {df['date'].min()} to {df['date'].max()}")

# Show average wait times per day — should match TRUE_PATTERNS roughly
print("\n📊 VERIFICATION - Average Wait Times:")
for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']:
    day_data = df[df['day_name'] == day]
    if len(day_data) > 0:
        avg = day_data['waiting_time_min'].mean()
        print(f"   {day}: {avg:.1f} min")

# Show hourly comparison between Monday (busiest) and Wednesday (calmest)
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