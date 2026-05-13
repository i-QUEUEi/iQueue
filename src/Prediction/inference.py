import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Preprocessing.features import FEATURES
from .constants import MONTE_CARLO_RUNS, RNG
from .context import avg_service_time, holiday_month_days, model, queue_maps, wait_maps
from .patterns import get_pattern_value


def get_actual_queue_length(day_name, month, week, hour):
    return get_pattern_value(queue_maps, day_name, month, week, hour, 10)


def get_actual_lag_features(day_name, month, week, hour, queue_length):
    prev_hour = hour - 1
    if prev_hour >= 8:
        prev_queue = get_pattern_value(queue_maps, day_name, month, week, prev_hour, queue_length)
        prev_wait = get_pattern_value(wait_maps, day_name, month, week, prev_hour, queue_length * 1.5)
    else:
        prev_queue = queue_length
        prev_wait = queue_length * 1.5
    return prev_queue, prev_wait


def get_month_features(target_date):
    month = target_date.month
    angle = 2 * np.pi * (month - 1) / 12
    month_sin = float(np.sin(angle))
    month_cos = float(np.cos(angle))
    days_in_month = pd.Timestamp(target_date).days_in_month
    is_end_of_month = 1 if target_date.day >= days_in_month - 2 else 0
    return month, month_sin, month_cos, is_end_of_month


def get_holiday_flags(target_date):
    if not holiday_month_days:
        return 0, 0
    is_holiday = 1 if (target_date.month, target_date.day) in holiday_month_days else 0
    next_day = target_date + pd.Timedelta(days=1)
    is_pre_holiday = 1 if (next_day.month, next_day.day) in holiday_month_days else 0
    return is_holiday, is_pre_holiday


def predict_wait_time(day_name, target_date, hour):
    day_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
    }
    day_of_week = day_map[day_name]
    week_of_month = (target_date.day - 1) // 7 + 1
    month, month_sin, month_cos, is_end_of_month = get_month_features(target_date)
    is_holiday, is_pre_holiday = get_holiday_flags(target_date)
    is_peak_day = 1 if day_name in ["Monday", "Friday"] else 0
    is_weekend = 1 if day_name == "Saturday" else 0

    if is_peak_day:
        is_peak_hour = 1 if hour in [9, 10, 11, 13, 14, 15] else 0
    else:
        is_peak_hour = 1 if hour in [9, 10, 14, 15] else 0

    queue_length = get_actual_queue_length(day_name, month, week_of_month, hour)
    lag_queue, lag_wait = get_actual_lag_features(day_name, month, week_of_month, hour, queue_length)

    features = [
        hour,
        day_of_week,
        week_of_month,
        month,
        month_sin,
        month_cos,
        is_end_of_month,
        is_holiday,
        is_pre_holiday,
        is_peak_day,
        queue_length,
        avg_service_time,
        is_weekend,
        is_peak_hour,
        lag_queue,
        lag_wait,
    ]

    X = pd.DataFrame([features], columns=FEATURES)
    wait_time = model.predict(X)[0]
    return round(wait_time, 1), queue_length


def predict_wait_time_monte_carlo(target_date, hour, runs=MONTE_CARLO_RUNS):
    day_name = target_date.strftime("%A")
    day_map = {
        "Monday": 0,
        "Tuesday": 1,
        "Wednesday": 2,
        "Thursday": 3,
        "Friday": 4,
        "Saturday": 5,
    }
    day_of_week = day_map[day_name]
    week_of_month = (target_date.day - 1) // 7 + 1
    month, month_sin, month_cos, is_end_of_month = get_month_features(target_date)
    is_holiday, is_pre_holiday = get_holiday_flags(target_date)
    is_peak_day = 1 if day_name in ["Monday", "Friday"] else 0
    is_weekend = 1 if day_name == "Saturday" else 0

    if is_peak_day:
        is_peak_hour = 1 if hour in [9, 10, 11, 13, 14, 15] else 0
    else:
        is_peak_hour = 1 if hour in [9, 10, 14, 15] else 0

    queue_base = get_actual_queue_length(day_name, month, week_of_month, hour)
    lag_queue_base, lag_wait_base = get_actual_lag_features(day_name, month, week_of_month, hour, queue_base)

    queue_samples = RNG.normal(queue_base, max(1.0, queue_base * 0.15), runs)
    queue_samples = np.clip(queue_samples, 1, None)

    service_samples = RNG.normal(avg_service_time, max(1.0, avg_service_time * 0.10), runs)
    service_samples = np.clip(service_samples, 5, None)

    lag_queue_samples = RNG.normal(lag_queue_base, max(1.0, lag_queue_base * 0.20), runs)
    lag_queue_samples = np.clip(lag_queue_samples, 1, None)

    lag_wait_samples = RNG.normal(lag_wait_base, max(1.5, lag_wait_base * 0.20), runs)
    lag_wait_samples = np.clip(lag_wait_samples, 5, None)

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
            "queue_length_at_arrival": queue_samples,
            "service_time_min": service_samples,
            "is_weekend": np.full(runs, is_weekend),
            "is_peak_hour": np.full(runs, is_peak_hour),
            "queue_length_lag1": lag_queue_samples,
            "waiting_time_lag1": lag_wait_samples,
        },
        columns=FEATURES,
    )

    wait_samples = model.predict(X)
    wait_samples = np.clip(wait_samples, 5, 90)

    return {
        "mean": round(float(np.mean(wait_samples)), 1),
        "p10": round(float(np.percentile(wait_samples, 10)), 1),
        "p50": round(float(np.percentile(wait_samples, 50)), 1),
        "p90": round(float(np.percentile(wait_samples, 90)), 1),
        "queue_mean": round(float(np.mean(queue_samples)), 1),
    }


def get_congestion_level(wait_time):
    if wait_time > 45:
        return "🔴 HIGH", "❌ AVOID - Very long queues (45+ min)"
    if wait_time > 25:
        return "🟡 MODERATE", "⚠️ CAUTION - Moderate wait (25-45 min)"
    return "🟢 LOW", "✅ GOOD - Short wait (<25 min)"
