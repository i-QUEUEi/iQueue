# Preprocessing — Study Script
### iQueue Preprocessing Pipeline | Full Walkthrough

---

## WHAT IS THIS FOLDER?

`src/Preprocessing/` is the **data preparation layer** of iQueue.

It takes the raw CSV (`synthetic_lto_cdo_queue_90days.csv`) — which already
has many columns — and does three things on top of it:

1. Adds one missing column: `week_of_month`
2. Re-calculates holiday flags from the **actual** Philippine holiday calendar
3. Removes any invalid rows (negative values, nulls)

The output is a clean DataFrame that the model can directly train on.

**It does NOT train any model.** That is `train_model.py`'s job.

---

## HOW IT CONNECTS TO THE PIPELINE

```
Data_.py
   ↓
synthetic_lto_cdo_queue_90days.csv  (raw, 20 columns)
   ↓
Preprocessing/
   calendar.py  →  loads holiday dates
   loader.py    →  reads CSV, adds week_of_month, recalculates holidays, cleans rows
   features.py  →  selects 16 model input columns (X) + target (y)
   preprocess.py → front door hub + standalone runner
   ↓
preprocessed_data.csv  (clean, 21 columns)
   ↓
train_model.py  →  model training
   ↓
predictions
```

- Preprocessing is the ONLY stage between raw data and training
- All changes happen **in-memory** — the raw CSV is never modified
- The preprocessed CSV is an optional saved snapshot, not required for training

---

## THE 4 FILES

```
Preprocessing/
│
├── preprocess.py   ← hub: re-exports everything + standalone runner
├── loader.py       ← reads CSV, cleans, engineers features
├── features.py     ← defines the 16 model features, builds feature rows
└── calendar.py     ← parses PH holiday calendar into a fast lookup set
```

Each file has exactly one responsibility. They connect like this:

```
calendar.py
   ↑ used by both ↑
loader.py        features.py
      ↑ both exported by ↑
          preprocess.py
```

---

## FILE 1 — calendar.py

**One job:** Read the Philippine holiday CSV and return a fast lookup set.

```python
def load_ph_holiday_month_days(calendar_path):
    ...
    return {(1,1), (4,9), (12,25), ...}  # set of (month, day) tuples
```

### Why a set?

Checking `(month, day) in holiday_set` happens thousands of times
across all rows. A **set** does this in O(1) — instant.
A list would scan every element each time — O(n) — much slower.

### The bug that was fixed

The holiday CSV stores **3 holidays per line** (left / center / right columns):

```
Jan 1: New Year's Day,,,,,,,,Apr 9: Day of Valor,,,,,,,,Sep 25: Mid-Autumn Festival
```

The original code used `re.search()` — it only found the **first match per line**.
So Sep 25, Nov 1, Nov 2, Nov 30, Dec 8, Dec 24, Dec 25, Dec 30, Dec 31
were all silently skipped.

**The fix:** switched to `re.findall()` — finds ALL matches on every line.

```python
# WRONG (old) — only finds "Jan 1:" and stops
match = re.search(r"\b([A-Za-z]{3})\s+(\d{1,2})\s*:\s*", line)

# CORRECT (fixed) — finds all 3 per line
matches = re.findall(r"\b([A-Za-z]{3})\s+(\d{1,2})\s*:\s*", line)
for month_name_raw, day_str in matches:
    ...
```

**Before fix:** 11 holidays detected (left column only)
**After fix:** 31 holidays detected (all 3 columns)

---

## FILE 2 — loader.py

**One job:** Load the raw CSV and return a clean, feature-rich DataFrame.

```python
def load_data(path):
    df = pd.read_csv(path)
    # ... 3 steps below ...
    return df
```

### Step 1 — Read and parse dates

```python
df = pd.read_csv(path)
df["date"] = pd.to_datetime(df["date"])  # "2026-01-05" → datetime(2026, 1, 5)
```

The CSV stores dates as strings. Converting to datetime objects lets
pandas compute things like `dt.month`, `dt.day`, `dt.days_in_month`.

### Step 2 — Engineer new columns

**month** (re-extracted from date)
```python
df["month"] = df["date"].dt.month
```
Raw integer 1–12.

**month_sin and month_cos** (cyclical month encoding)
```python
month_angle = 2 * np.pi * (df["month"] - 1) / 12
df["month_sin"] = np.sin(month_angle)
df["month_cos"] = np.cos(month_angle)
```
The raw CSV already has these, but loader.py re-computes them from
the parsed `month` column to ensure they're mathematically correct.

See DATA_GENERATION.MD for the full explanation of why sin/cos encoding
is used instead of raw month numbers.

**is_end_of_month**
```python
df["is_end_of_month"] = (df["date"].dt.day >= (df["date"].dt.days_in_month - 2)).astype(int)
```
`days_in_month` gives the total days in that month (28/29/30/31).
Last 3 days = `day >= last_day - 2`.

| Month | Last day | Threshold | Rush days       |
|-------|----------|-----------|-----------------|
| Jan   | 31       | 29        | Jan 29, 30, 31  |
| Feb   | 28       | 26        | Feb 26, 27, 28  |
| Apr   | 30       | 28        | Apr 28, 29, 30  |

**is_holiday and is_pre_holiday** (re-calculated from real calendar)
```python
holiday_md = load_ph_holiday_month_days(HOLIDAY_CALENDAR_PATH)
df["is_holiday"]     = df["date"].apply(lambda d: 1 if (d.month, d.day) in holiday_md else 0)
df["is_pre_holiday"] = df["date"].apply(lambda d: 1 if (next_day.month, next_day.day) in holiday_md else 0)
```

The raw CSV already has `is_holiday` and `is_pre_holiday` from Data_.py,
but loader.py **overwrites them** using the authoritative calendar file.
This ensures holiday flags are accurate even if the synthetic generator
had its own (potentially different) holiday logic.

**week_of_month** ← the only NEW column preprocessing adds
```python
df["week_of_month"] = df["date"].dt.day.apply(lambda d: (d - 1) // 7 + 1)
```

| Day range | week_of_month |
|-----------|---------------|
| Days 1–7  | 1             |
| Days 8–14 | 2             |
| Days 15–21| 3             |
| Days 22–28| 4             |
| Days 29+  | 5             |

This tells the model which week within the month it is. Week 1 of
January and week 1 of December behave differently from week 4 — people
rush before month-end deadlines in week 4+.

### Step 3 — Remove invalid rows

```python
df = df[df["waiting_time_min"] >= 0]        # negative wait times are impossible
df = df[df["queue_length_at_arrival"] >= 0]  # negative queue lengths are impossible
df = df.dropna()                             # remove rows with any missing values
```

Three filters in order:
1. **Negative wait times** — physically impossible, must be a generation error
2. **Negative queue lengths** — same
3. **Null values** — any row missing any column is dropped

In the current dataset, no rows are removed (21,428 in → 21,428 out).
These are safety nets for real or future data.

### What the print summary shows

After loading, loader.py prints:
```
[OK] Loaded 21428 records
[INFO] Date range: 2026-01-01 to 2026-11-03

[INFO] Data distribution by day:
   Monday: 4304 records
   Tuesday: 4291 records
   ...
```

This lets you verify at a glance that the data loaded correctly
before training starts.

---

## FILE 3 — features.py

**One job:** Define exactly which 16 columns the model sees, and provide
functions to extract them from a DataFrame.

### The 16 features

```python
FEATURES = [
    "hour",                    # What time of day (8–16)
    "day_of_week",             # Monday=0 through Saturday=5
    "week_of_month",           # Week 1, 2, 3, 4, or 5 within the month
    "month",                   # January=1 through December=12
    "month_sin",               # Cyclical month encoding (sine component)
    "month_cos",               # Cyclical month encoding (cosine component)
    "is_end_of_month",         # 1 if within last 3 days of month, else 0
    "is_holiday",              # 1 if this is a Philippine holiday, else 0
    "is_pre_holiday",          # 1 if tomorrow is a holiday, else 0
    "is_peak_day",             # 1 if Monday or Friday (busiest days), else 0
    "queue_length_at_arrival", # How many people are already in line when you arrive
    "service_time_min",        # How long the current transaction takes (minutes)
    "is_weekend",              # 1 if Saturday, else 0
    "is_peak_hour",            # 1 if during peak hours (9-11am, 2-3pm), else 0
    "queue_length_lag1",       # Queue length of the PREVIOUS transaction (memory)
    "waiting_time_lag1",       # Wait time of the PREVIOUS transaction (memory)
]
```

These are deliberately chosen to NOT include `waiting_time_min`
(the target), `date`, `arrival_time`, `day_name`, or `total_time_in_system_min`.
The model is only allowed to see what would be available at prediction time.

### get_features(df)

```python
def get_features(df):
    X = df[FEATURES]              # 16-column input matrix
    y = df["waiting_time_min"]    # target: what we're predicting
    return X, y, FEATURES
```

Returns three things:
- **X** — the 16-column DataFrame the model trains on
- **y** — the target column (wait times in minutes)
- **FEATURES** — the list of names (for reference and column ordering)

### build_feature_dataframe(records)

Used during **prediction** (not training). When a user asks
"What's the wait at 9am on Monday Jan 5?", this function builds
a one-row DataFrame with all 16 features filled in — matching
the exact format the trained model expects.

```python
# Input: a dict like {"date": "2026-03-15", "hour": 9, "queue_length_at_arrival": 10}
# Output: DataFrame with all 16 features computed
```

It derives features from the date (month, week_of_month, cyclical encoding,
holiday flags) and uses defaults for anything not provided
(e.g., `service_time_min` defaults to 35 min).

---

## FILE 4 — preprocess.py

**Two jobs:**

### Job 1 — The front door (import hub)

Other parts of the codebase only need one import statement:

```python
from Preprocessing.preprocess import load_data, get_features
```

Instead of three separate imports from three files.
preprocess.py re-exports everything from loader, features, and calendar.

```python
__all__ = [
    "FEATURES",
    "build_feature_dataframe",
    "get_features",
    "load_data",
    "load_ph_holiday_month_days",
]
```

### Job 2 — Standalone runner

When run directly (`python -m Preprocessing.preprocess` from `src/`),
the `if __name__ == "__main__"` block fires:

```
Step 1 — Resolve paths
         Walks 2 levels up from preprocess.py to find the iQueue/ root

Step 2 — Check raw CSV exists
         If missing → print error, exit

Step 3 — load_data(DATA_PATH)
         Runs the full preprocessing pipeline in memory

Step 4 — get_features(df)
         Extracts X and y for printing summary stats

Step 5 — Print summary
         Total rows, columns, feature count, y stats, null count

Step 6 — Save to disk
         Creates data/Preprocessed/ if it doesn't exist
         Writes preprocessed_data.csv
```

### The try/except import trick

```python
try:
    from .calendar import load_ph_holiday_month_days   # package mode
    ...
except ImportError:
    from Preprocessing.calendar import load_ph_holiday_month_days  # direct-run mode
```

Python's import system works differently depending on how the file is used:
- **Imported as a package** → relative imports (`.calendar`) work
- **Run directly** → relative imports fail, absolute imports are used instead

The `sys.path` patch at the top ensures `src/` is on the path before
the import block runs, so absolute imports always resolve correctly.

---

## RAW vs PREPROCESSED CSV

| | `synthetic_lto_cdo_queue_90days.csv` | `preprocessed_data.csv` |
|---|---|---|
| Columns | 20 | 21 |
| Rows | 21,428 | 21,428 |
| Column added | — | `week_of_month` |
| `is_holiday` | From Data_.py generator | Re-computed from real calendar |
| `is_pre_holiday` | From Data_.py generator | Re-computed from real calendar |
| Bad rows removed | No | Yes (none in this dataset) |

The 21 columns break down as:
- **16** → model input features (the `FEATURES` list)
- **1** → `waiting_time_min` → the target `y` (what the model predicts)
- **4** → metadata only (`date`, `arrival_time`, `day_name`, `total_time_in_system_min`)

---

## HOW TRAINING USES PREPROCESSING

`train_model.py` checks if the preprocessed CSV already exists:

```python
if PREPROCESSED_PATH.exists():
    # Fast path — load the already-cleaned CSV directly
    df = pd.read_csv(PREPROCESSED_PATH)
    df["date"] = pd.to_datetime(df["date"])
else:
    # Slow path — run the full preprocessing pipeline from raw CSV
    df = load_data(DATA_PATH)
```

This means:
- **First run** → preprocessing runs, then training
- **Subsequent runs** → training skips preprocessing and loads the cached CSV

---

## HOW main.py ORCHESTRATES PREPROCESSING

`main.py` runs all three pipeline stages in order:

```
[1/3] python -m Preprocessing.preprocess
      → runs preprocessing, saves preprocessed_data.csv

[2/3] python src/model_implementation/train_model.py
      → loads preprocessed_data.csv, trains model

[3/3] python src/Prediction/predict.py
      → loads trained model, runs CLI prediction
```

Preprocessing always runs first in the full pipeline so the cached
CSV is always fresh before training begins.

---

## SUMMARY — What each file does for the model

| File | What it provides |
|---|---|
| `calendar.py` | A fast set of `(month, day)` holiday tuples for O(1) lookups |
| `loader.py` | A clean DataFrame with `week_of_month` added and holiday flags corrected |
| `features.py` | The exact 16-column X matrix and target y the model trains on |
| `preprocess.py` | A single import point + the ability to run preprocessing standalone |

**The pattern across all of them:**
Each file does exactly one thing and delegates everything else.
`preprocess.py` doesn't load data — that's `loader.py`.
`loader.py` doesn't know about features — that's `features.py`.
`features.py` doesn't know about holidays — that's `calendar.py`.

---

*End of study script — Preprocessing | iQueue data preparation pipeline*
