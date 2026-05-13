from pathlib import Path
import sys

import joblib
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Preprocessing.calendar import load_ph_holiday_month_days
from .patterns import build_pattern_maps, get_pattern_value

BASE_DIR = Path(__file__).resolve().parents[2]
MODEL_PATH = BASE_DIR / "models" / "queue_model.pkl"
DATA_PATH = BASE_DIR / "data" / "synthetic_lto_cdo_queue_90days.csv"
HOLIDAY_CALENDAR_PATH = (
    BASE_DIR / "data" / "2026-calendar-with-holidays-portrait-sunday-start-en-ph.csv"
)

model = joblib.load(MODEL_PATH)

print("📊 Calculating actual queue patterns from training data...")

df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df["week_of_month"] = df["date"].dt.day.apply(lambda d: (d - 1) // 7 + 1)
df["month"] = df["date"].dt.month

queue_maps = build_pattern_maps(df, "queue_length_at_arrival")
wait_maps = build_pattern_maps(df, "waiting_time_min")

print("\n✅ Loaded date-aware queue patterns (12 months × 4 weeks × 6 days × 9 hours):")
print(
    "   Monday Wk1 9am avg queue: {} people".format(
        round(get_pattern_value(queue_maps, "Monday", 1, 1, 9, 5), 1)
    )
)
print(
    "   Monday Wk4 9am avg queue: {} people".format(
        round(get_pattern_value(queue_maps, "Monday", 1, 4, 9, 5), 1)
    )
)
print(
    "   Wednesday Wk1 9am avg queue: {} people".format(
        round(get_pattern_value(queue_maps, "Wednesday", 1, 1, 9, 5), 1)
    )
)
print(
    "   Wednesday Wk4 9am avg queue: {} people".format(
        round(get_pattern_value(queue_maps, "Wednesday", 1, 4, 9, 5), 1)
    )
)

holiday_month_days = load_ph_holiday_month_days(HOLIDAY_CALENDAR_PATH)
avg_service_time = df["service_time_min"].mean()
