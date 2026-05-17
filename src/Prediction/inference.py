"""Inference engine — builds feature rows and runs the ML model to get predictions.

This module contains the core prediction logic:
- predict_wait_time(): Single-point prediction for a specific day/hour
- predict_wait_time_monte_carlo(): 1,000 simulated predictions with random
  variation, producing a confidence range (P10 to P90)
- Helper functions to compute feature values (month encoding, holiday flags, etc.)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Ensure src/ is importable
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Preprocessing.features import FEATURES
from .constants import MONTE_CARLO_RUNS, RNG
from .context import avg_service_time, holiday_month_days, model, queue_maps, wait_maps
from .patterns import get_pattern_value


def get_actual_queue_length(day_name, month, week, hour):
    """Look up the historical average queue length for this day/month/week/hour.

    Falls back through 4 levels of specificity (see patterns.py).
    Default value of 10 if no historical data exists at any level.
    """
    return get_pattern_value(queue_maps, day_name, month, week, hour, 10)


def get_actual_lag_features(day_name, month, week, hour, queue_length):
    """Estimate what the queue/wait looked like ONE HOUR AGO (lag features).

    For hours 9–16: looks up the actual historical value for the previous hour.
    For 8am (first hour): estimates based on current queue length since
    there's no "previous hour" data.

    Args:
        day_name: Day of the week.
        month, week, hour: Time coordinates.
        queue_length: Current queue length (used as fallback for 8am).

    Returns:
        (prev_queue, prev_wait): Estimated queue length and wait time
        from the previous hour.
    """
    prev_hour = hour - 1
    if prev_hour >= 8:
        # Previous hour exists — look up historical values
        prev_queue = get_pattern_value(queue_maps, day_name, month, week, prev_hour, queue_length)
        prev_wait = get_pattern_value(wait_maps, day_name, month, week, prev_hour, queue_length * 1.5)
    else:
        # 8am — no previous hour, estimate from current queue
        prev_queue = queue_length
        prev_wait = queue_length * 1.5  # Rough estimate: wait ≈ 1.5× queue length
    return prev_queue, prev_wait


def get_month_features(target_date):
    """Compute all month-related features for a given date.

    Returns:
        month: Month number (1–12)
        month_sin: Cyclical sine encoding
        month_cos: Cyclical cosine encoding
        is_end_of_month: 1 if within last 3 days of month, else 0
    """
    month = target_date.month
    angle = 2 * np.pi * (month - 1) / 12  # Map month onto a circle
    month_sin = float(np.sin(angle))
    month_cos = float(np.cos(angle))
    days_in_month = pd.Timestamp(target_date).days_in_month
    is_end_of_month = 1 if target_date.day >= days_in_month - 2 else 0
    return month, month_sin, month_cos, is_end_of_month


def get_holiday_flags(target_date):
    """Check if a date is a holiday or the day before a holiday.

    Returns:
        is_holiday: 1 if this date is a Philippine holiday, else 0
        is_pre_holiday: 1 if tomorrow is a holiday, else 0
    """
    if not holiday_month_days:
        return 0, 0  # No holiday data available
    is_holiday = 1 if (target_date.month, target_date.day) in holiday_month_days else 0
    next_day = target_date + pd.Timedelta(days=1)
    is_pre_holiday = 1 if (next_day.month, next_day.day) in holiday_month_days else 0
    return is_holiday, is_pre_holiday


def get_holiday_name(target_date):
    """Return the name of the holiday on a given date, or None if not a holiday.

    Reads the calendar CSV and matches the date to a holiday entry like
    "Jan 1: New Year's Day", returning the name part ("New Year's Day").

    Args:
        target_date: A pandas Timestamp or date-like object.

    Returns:
        The holiday name string, e.g. "Christmas Day",
        or "Philippine Holiday" as a fallback if the name can't be parsed.
    """
    import re
    from .context import HOLIDAY_CALENDAR_PATH
    from Preprocessing.calendar import MONTH_MAP

    month = target_date.month
    day = target_date.day

    # Reverse-lookup: month number → 3-letter abbreviation
    month_abbr = {v: k for k, v in MONTH_MAP.items()}.get(month, "")

    if not HOLIDAY_CALENDAR_PATH.exists():
        return "Philippine Holiday"

    text = HOLIDAY_CALENDAR_PATH.read_text(encoding="utf-8", errors="ignore")
    for line in text.splitlines():
        # Match entries like "Jan 1: New Year's Day" anywhere in the line
        matches = re.findall(r"\b([A-Za-z]{3})\s+(\d{1,2})\s*:\s*([^,]+)", line)
        for m_abbr, m_day, m_name in matches:
            if m_abbr.title() == month_abbr and int(m_day) == day:
                return m_name.strip()

    return "Philippine Holiday"  # Fallback if name not found in file


def predict_wait_time(day_name, target_date, hour):
    """Make a single-point prediction for a specific day/hour combination.

    Assembles all 16 features and passes them to the trained model.

    Args:
        day_name: e.g., "Monday"
        target_date: The date to predict for
        hour: Hour of day (8–16)

    Returns:
        (predicted_wait_time, queue_length): Tuple of predicted wait in minutes
        and the estimated queue length at arrival.
    """
    # Map day name to numeric value
    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5,
    }
    day_of_week = day_map[day_name]
    week_of_month = (target_date.day - 1) // 7 + 1

    # Compute all feature components
    month, month_sin, month_cos, is_end_of_month = get_month_features(target_date)
    is_holiday, is_pre_holiday = get_holiday_flags(target_date)
    is_peak_day = 1 if day_name in ["Monday", "Friday"] else 0
    is_weekend = 1 if day_name == "Saturday" else 0

    # Peak hours differ between peak days and non-peak days
    if is_peak_day:
        is_peak_hour = 1 if hour in [9, 10, 11, 13, 14, 15] else 0
    else:
        is_peak_hour = 1 if hour in [9, 10, 14, 15] else 0

    # Look up historical queue and lag values from pattern maps
    queue_length = get_actual_queue_length(day_name, month, week_of_month, hour)
    lag_queue, lag_wait = get_actual_lag_features(day_name, month, week_of_month, hour, queue_length)

    # Assemble all 16 features in the exact order the model expects
    features = [
        hour, day_of_week, week_of_month, month,
        month_sin, month_cos, is_end_of_month,
        is_holiday, is_pre_holiday, is_peak_day,
        queue_length, avg_service_time,
        is_weekend, is_peak_hour,
        lag_queue, lag_wait,
    ]

    # Create a DataFrame and run the prediction
    X = pd.DataFrame([features], columns=FEATURES)
    wait_time = model.predict(X)[0]
    return round(wait_time, 1), queue_length


def predict_wait_time_monte_carlo(target_date, hour, runs=MONTE_CARLO_RUNS):
    """Run 1,000 predictions with random variation to get a confidence range.

    Instead of one prediction, we run `runs` (default 1,000) predictions,
    each with slightly different queue lengths and lag values sampled from
    normal distributions. This produces a RANGE of likely outcomes.

    The output includes:
    - mean: Average predicted wait time
    - p10: Optimistic estimate (90% chance wait is longer than this)
    - p50: Median estimate
    - p90: Pessimistic estimate (90% chance wait is shorter than this)

    Args:
        target_date: The date to predict for.
        hour: Hour of day (8–16).
        runs: Number of Monte Carlo simulation runs (default: 1000).

    Returns:
        Dictionary with mean, p10, p50, p90, and queue_mean.
    """
    day_name = target_date.strftime("%A")
    day_map = {
        "Monday": 0, "Tuesday": 1, "Wednesday": 2,
        "Thursday": 3, "Friday": 4, "Saturday": 5,
    }
    day_of_week = day_map[day_name]
    week_of_month = (target_date.day - 1) // 7 + 1

    # Compute deterministic features (same for all 1,000 runs)
    month, month_sin, month_cos, is_end_of_month = get_month_features(target_date)
    is_holiday, is_pre_holiday = get_holiday_flags(target_date)
    is_peak_day = 1 if day_name in ["Monday", "Friday"] else 0
    is_weekend = 1 if day_name == "Saturday" else 0

    if is_peak_day:
        is_peak_hour = 1 if hour in [9, 10, 11, 13, 14, 15] else 0
    else:
        is_peak_hour = 1 if hour in [9, 10, 14, 15] else 0

    # Get historical base values for the stochastic features
    queue_base = get_actual_queue_length(day_name, month, week_of_month, hour)
    lag_queue_base, lag_wait_base = get_actual_lag_features(day_name, month, week_of_month, hour, queue_base)

    # ===== Generate random samples around the base values =====
    # Each sample represents a possible "what if?" scenario

    # Queue length: ±15% random variation around historical average
    queue_samples = RNG.normal(queue_base, max(1.0, queue_base * 0.15), runs)
    queue_samples = np.clip(queue_samples, 1, None)  # At least 1 person in queue

    # Service time: ±10% variation around global average
    service_samples = RNG.normal(avg_service_time, max(1.0, avg_service_time * 0.10), runs)
    service_samples = np.clip(service_samples, 5, None)  # At least 5 min service

    # Lag queue: ±20% variation (more uncertain — it's an estimate of the past)
    lag_queue_samples = RNG.normal(lag_queue_base, max(1.0, lag_queue_base * 0.20), runs)
    lag_queue_samples = np.clip(lag_queue_samples, 1, None)

    # Lag wait: ±20% variation
    lag_wait_samples = RNG.normal(lag_wait_base, max(1.5, lag_wait_base * 0.20), runs)
    lag_wait_samples = np.clip(lag_wait_samples, 5, None)

    # ===== Build a DataFrame with 1,000 rows (one per simulation run) =====
    # Deterministic features use np.full() (same value for all runs)
    # Stochastic features use the random samples
    X = pd.DataFrame(
        {
            "hour": np.full(runs, hour),
            "day_of_week": np.full(runs, day_of_week),
            "week_of_month": np.full(runs, week_of_month),
            "month": np.full(runs, month),
            "month_sin": np.full(runs, month_sin),
            "month_cos": np.full(runs, month_cos),
            "is_end_of_month": np.full(runs, is_end_of_month),
            "is_holiday": np.full(runs, is_holiday),
            "is_pre_holiday": np.full(runs, is_pre_holiday),
            "is_peak_day": np.full(runs, is_peak_day),
            "queue_length_at_arrival": queue_samples,     # Random variation
            "service_time_min": service_samples,           # Random variation
            "is_weekend": np.full(runs, is_weekend),
            "is_peak_hour": np.full(runs, is_peak_hour),
            "queue_length_lag1": lag_queue_samples,         # Random variation
            "waiting_time_lag1": lag_wait_samples,          # Random variation
        },
        columns=FEATURES,
    )

    # Run all 1,000 predictions in one batch (much faster than 1,000 individual calls)
    wait_samples = model.predict(X)
    wait_samples = np.clip(wait_samples, 5, 90)  # Clamp to realistic range [5, 90] min

    # ===== Compute summary statistics from the 1,000 results =====
    return {
        "mean": round(float(np.mean(wait_samples)), 1),          # Average prediction
        "p10": round(float(np.percentile(wait_samples, 10)), 1), # Optimistic (10th percentile)
        "p50": round(float(np.percentile(wait_samples, 50)), 1), # Median
        "p90": round(float(np.percentile(wait_samples, 90)), 1), # Pessimistic (90th percentile)
        "queue_mean": round(float(np.mean(queue_samples)), 1),   # Average simulated queue
    }


def get_congestion_level(wait_time):
    """Convert a predicted wait time into a human-readable congestion level.

    Args:
        wait_time: Predicted wait time in minutes.

    Returns:
        (level_emoji, recommendation): Tuple of colored label and action advice.
    """
    if wait_time > 45:
        return "🔴 HIGH", "❌ AVOID - Very long queues (45+ min)"
    if wait_time > 25:
        return "🟡 MODERATE", "⚠️ CAUTION - Moderate wait (25-45 min)"
    return "🟢 LOW", "✅ GOOD - Short wait (<25 min)"
