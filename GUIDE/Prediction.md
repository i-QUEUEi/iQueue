# Prediction — Study Script
### iQueue Prediction System | Full Walkthrough

---

## WHAT IS THIS SECTION?

`src/Prediction/` is the **user-facing layer** of iQueue.

It takes the trained model (`queue_model.pkl`) and answers one question:
**"How long will I wait at LTO CDO today?"**

It does this three ways:
1. Weekly forecast — all 6 working days of a chosen week
2. Daily forecast — hour-by-hour breakdown for one specific date
3. Best time finder — which hour has the shortest wait

**It does NOT train anything.** That is `train_model.py`'s job.
**It does NOT preprocess data.** That is `Preprocessing/`'s job.
**It does NOT evaluate models.** That is `Evaluation/`'s job.

---

## HOW IT CONNECTS TO THE PIPELINE

```
queue_model.pkl  (trained by Model Training — see Model_Training.md)
   +
synthetic_lto_cdo_queue_90days.csv  (used to build historical patterns only)
   ↓
context.py     ← loads model + builds pattern tables at startup
patterns.py    ← defines the 4-level historical lookup
inference.py   ← assembles 16 features, runs model, applies Monte Carlo
cli.py         ← the menu the user actually sees and interacts with
constants.py   ← MONTE_CARLO_RUNS and RNG seed
predict.py     ← entry point: patches sys.path, imports, calls main()
```

---

## THE 6 FILES

```
Prediction/
│
├── predict.py     ← entry point, launched by main.py
├── constants.py   ← MONTE_CARLO_RUNS = 1000, RNG seed = 42
├── context.py     ← startup: loads model + pattern tables (runs at import)
├── patterns.py    ← build_pattern_maps() + get_pattern_value()
├── inference.py   ← predict_wait_time(), Monte Carlo, get_congestion_level()
└── cli.py         ← interactive menu with 3 forecast views
```

---

## THE CONSTANTS — constants.py

```python
MONTE_CARLO_RUNS = 1000
RNG = np.random.default_rng(42)
```

**MONTE_CARLO_RUNS = 1000**
Each prediction runs the model 1,000 times with slightly different
inputs, producing a range of outcomes instead of one number.
"Your wait is probably 28–45 minutes" is more useful than "32 minutes."

**RNG = np.random.default_rng(42)**
A modern NumPy random number generator with seed 42.
The seed ensures the same 1,000 random samples are generated
every time — so predictions are reproducible.

Why `default_rng` instead of `np.random.seed(42)`?
- `default_rng` is thread-safe (can run in parallel safely)
- Not global state — it's a generator object that you pass around
- Follows modern NumPy best practices

---

## STARTUP — context.py

`context.py` runs **at import time** — the moment any other file does
`from .context import model`. This is deliberate: everything is loaded
once when predict.py starts, not on every prediction.

### Step 1 — Load the trained model

```python
model = joblib.load(MODEL_PATH)
# → loads queue_model.pkl
# → full Random Forest: all 500 trees, all splits, all leaf values
# → ready to call model.predict(X) immediately
```

`joblib.load()` deserializes the `.pkl` file back into a fully trained
sklearn RandomForestRegressor object. This takes ~1–2 seconds.

### Step 2 — Read the CSV and build pattern tables

```python
df = pd.read_csv(DATA_PATH)
df["date"] = pd.to_datetime(df["date"])
df["week_of_month"] = df["date"].dt.day.apply(lambda d: (d - 1) // 7 + 1)
df["month"] = df["date"].dt.month

queue_maps = build_pattern_maps(df, "queue_length_at_arrival")
wait_maps  = build_pattern_maps(df, "waiting_time_min")
```

The raw CSV is read to compute historical averages — NOT for retraining.
Two pattern tables are built: one for queue lengths, one for wait times.
These are used to fill in the "queue_length_at_arrival" and lag features
when making predictions (since we don't know the real queue in advance).

### Step 3 — Load the holiday calendar

```python
holiday_month_days = load_ph_holiday_month_days(HOLIDAY_CALENDAR_PATH)
# → returns {(1,1), (4,9), (12,25), ...}  — set of (month, day) tuples
```

Same calendar module as `Preprocessing/` (see Preprocessing.md).
Used for the holiday guard in cli.py and inference.py.

### Step 4 — Compute global average service time

```python
avg_service_time = df["service_time_min"].mean()
# → e.g., 38.7 min — used as default service_time feature in predictions
```

---

## THE PATTERN TABLES — patterns.py

The pattern tables are the answer to: **"What queue length should I use
in the model input when predicting the future?"**

The model was trained on real queue lengths. When predicting, we don't
know the real queue yet — so we look up the **historical average** for
that exact day/month/week/hour.

### build_pattern_maps(df, value_col)

Builds **4 nested lookup tables** at different levels of specificity:

```
Level 4 (most specific):  day + month + week_of_month + hour
Level 3:                  day + month + hour
Level 2:                  day + week_of_month + hour
Level 1 (broadest):       day + hour
```

**Why 4 levels?**
The most specific level (Monday, January, Week 2, 9am) might have
very few data points — or none if that combination never appeared.
Having 4 fallback levels guarantees we always return something.

**Example — Monday, January, Week 4, 9am:**
```
Level 4: avg queue for "Mon + Jan + Wk4 + 9am"  → e.g., 31.2 people ✅ use this
```

**Example — Monday, December, Week 5, 9am:**
```
Level 4: "Mon + Dec + Wk5 + 9am" → NaN (no data)
Level 3: "Mon + Dec + 9am" → 27.8 people ✅ use this
```

### get_pattern_value(pattern_maps, day_name, month, week, hour, default)

```python
# Try Level 4 → Level 3 → Level 2 → Level 1 → default
value = pattern_maps["day_month_week_hour"][day][month][week][hour]
if not pd.isna(value): return float(value)

value = pattern_maps["day_month_hour"][day][month][hour]
if not pd.isna(value): return float(value)

value = pattern_maps["day_week_hour"][day][week][hour]
if not pd.isna(value): return float(value)

value = pattern_maps["day_hour"][day][hour]
if not pd.isna(value): return float(value)

return float(default_value)  # absolute fallback
```

**The key insight:** Predictions are not generic. "Monday 9am" in week 4
is treated differently from "Monday 9am" in week 1. Month-end weeks are
busier — the pattern table captures that automatically from the training data.

---

## THE INFERENCE ENGINE — inference.py

`inference.py` is where the ML model actually runs. It has 3 key functions:

### 1 — predict_wait_time() — single prediction

Used internally for comparison. Builds all 16 features and calls the model once.

```python
def predict_wait_time(day_name, target_date, hour):
    # 1. Compute all deterministic features from the date
    month, month_sin, month_cos, is_end_of_month = get_month_features(target_date)
    is_holiday, is_pre_holiday = get_holiday_flags(target_date)
    is_peak_day  = 1 if day_name in ["Monday", "Friday"] else 0
    is_weekend   = 1 if day_name == "Saturday" else 0
    is_peak_hour = 1 if hour in [9, 10, 11, 13, 14, 15] else 0  # peak days
                   1 if hour in [9, 10, 14, 15] else 0            # other days

    # 2. Look up historical patterns for stochastic features
    queue_length = get_actual_queue_length(day_name, month, week_of_month, hour)
    lag_queue, lag_wait = get_actual_lag_features(day_name, month, week_of_month, hour, queue_length)

    # 3. Pack all 16 features into a DataFrame in the exact order the model expects
    X = pd.DataFrame([features], columns=FEATURES)

    # 4. Run the model
    wait_time = model.predict(X)[0]
    return round(wait_time, 1), queue_length
```

The 16 features are assembled in the EXACT same order as `FEATURES` from
`Preprocessing/features.py` (see Preprocessing.md). This is critical —
if any column is in the wrong position, the model sees incorrect data.

### The lag features for prediction

```python
def get_actual_lag_features(day_name, month, week, hour, queue_length):
    prev_hour = hour - 1
    if prev_hour >= 8:
        # Look up historical values for the previous hour
        prev_queue = get_pattern_value(queue_maps, day_name, month, week, prev_hour, queue_length)
        prev_wait  = get_pattern_value(wait_maps,  day_name, month, week, prev_hour, queue_length * 1.5)
    else:
        # 8am — no previous hour exists
        prev_queue = queue_length
        prev_wait  = queue_length * 1.5
    return prev_queue, prev_wait
```

The lag features (`queue_length_lag1`, `waiting_time_lag1`) represent
what the previous person experienced. For training, these were computed
from the actual previous row. For prediction, we estimate them from
historical pattern averages for the previous hour.

**Why queue_length × 1.5 for 8am?**
At 8am there's no "previous hour." The office just opened. The rough
heuristic `wait ≈ 1.5 × queue_length` (e.g., 10 people → ~15 min wait)
approximates a standing queue being processed from zero.

### 2 — predict_wait_time_monte_carlo() — 1,000 simulations

This is what cli.py actually calls. Instead of one prediction, it runs
1,000 predictions with randomized inputs and returns a confidence range.

**Why Monte Carlo?**

Queue prediction has inherent uncertainty. The model says "32 minutes"
but it could be 20 or 50 depending on how many staff are working,
whether someone brings a complicated case, etc.

Instead of pretending the prediction is precise, Monte Carlo says:
"Given realistic random variation, 80% of the time you'll wait
between P10 and P90 minutes."

**How the random variation works:**

```python
# Queue length: ±15% variation around historical average
queue_samples = RNG.normal(queue_base, queue_base * 0.15, 1000)
queue_samples = np.clip(queue_samples, 1, None)

# Service time: ±10% variation
service_samples = RNG.normal(avg_service_time, avg_service_time * 0.10, 1000)
service_samples = np.clip(service_samples, 5, None)

# Lag features: ±20% variation (more uncertain — it's an estimate)
lag_queue_samples = RNG.normal(lag_queue_base, lag_queue_base * 0.20, 1000)
lag_wait_samples  = RNG.normal(lag_wait_base,  lag_wait_base  * 0.20, 1000)
```

**Why normal distribution?**
Queue arrivals follow a normal (Gaussian) distribution around the mean
in practice. The ±% is the standard deviation — most samples land close
to the base, with fewer very high or very low outliers.

**The 1,000-row batch prediction:**

```python
X = pd.DataFrame(
    {
        "hour":                    np.full(1000, hour),       # same for all runs
        "queue_length_at_arrival": queue_samples,             # random each run
        "service_time_min":        service_samples,           # random each run
        "queue_length_lag1":       lag_queue_samples,         # random each run
        "waiting_time_lag1":       lag_wait_samples,          # random each run
        ...                                                   # rest fixed
    },
    columns=FEATURES
)
wait_samples = model.predict(X)       # 1,000 predictions in one call
wait_samples = np.clip(wait_samples, 5, 90)  # clamp to [5, 90] min
```

Running 1,000 predictions as a batch is MUCH faster than 1,000 individual
`model.predict()` calls because sklearn processes the entire matrix at once.

**The output:**

```python
return {
    "mean": 32.4,  # Average of all 1,000 predictions
    "p10":  22.1,  # 10th percentile — optimistic estimate
    "p50":  31.8,  # Median
    "p90":  44.7,  # 90th percentile — pessimistic estimate
    "queue_mean": 18.3,  # Average of the 1,000 simulated queue lengths
}
```

**What P10-P90 means in plain English:**
```
P10 = 22 min  →  Best case: 10% chance you wait less than this
P90 = 45 min  →  Worst case: 10% chance you wait more than this
mean = 32 min →  Most likely outcome
```

### 3 — get_congestion_level()

```python
if wait_time > 45:  return "🔴 HIGH", "❌ AVOID - Very long queues (45+ min)"
if wait_time > 25:  return "🟡 MODERATE", "⚠️ CAUTION - Moderate wait (25-45 min)"
                    return "🟢 LOW", "✅ GOOD - Short wait (<25 min)"
```

Three thresholds. Gives every prediction a simple traffic-light label
so users don't have to interpret raw minutes.

### 4 — get_holiday_flags() and get_holiday_name()

```python
def get_holiday_flags(target_date):
    is_holiday = 1 if (target_date.month, target_date.day) in holiday_month_days else 0
    next_day = target_date + pd.Timedelta(days=1)
    is_pre_holiday = 1 if (next_day.month, next_day.day) in holiday_month_days else 0
    return is_holiday, is_pre_holiday
```

Used both as a feature in the model AND as a guard in cli.py.
When `is_holiday = 1`, the CLI shows "CLOSED" instead of predictions.

```python
def get_holiday_name(target_date):
    # Reads the calendar CSV and regex-matches "Jan 1: New Year's Day"
    # Returns the name part: "New Year's Day"
```

---

## THE CLI — cli.py

`cli.py` is the terminal interface the user actually sees. It has 3 views
and a date validator, all wrapped in a `while True` menu loop.

### The Holiday Guard

Every forecast function starts with this check BEFORE running any predictions:

```python
is_hol, _ = get_holiday_flags(target_date)
if is_hol:
    holiday_name = get_holiday_name(target_date)
    print(f"⛔ LTO CDO IS CLOSED — {holiday_name}")
    return  # ← exit early, no predictions made
```

Without this guard, the model would still run and produce predictions
for a day the office is closed — meaningless output. The guard ensures
users get a clear "CLOSED" message with the holiday name instead.

### View 1 — display_weekly_forecast()

Shows all 6 working days (Mon–Sat) of the chosen week:

```
Week of May 19, 2026 (Week 3 of month)

Monday (May 19):
  Overall: 51 min average (🔴 HIGH)
  Best:  08:00 (28 min, P10-P90: 22-35)
  Worst: 11:00 (63 min, P10-P90: 55-72)

Tuesday (May 20):   → if holiday: ⛔ CLOSED — Rizal Day
Wednesday (May 21):
  Overall: 24 min average (🟢 LOW)
  ...
```

**How it calculates "the Monday of this week":**
```python
week_start = target_date - pd.Timedelta(days=target_date.weekday())
# weekday() returns 0=Monday, 1=Tuesday... so subtracting it always gives Monday
```

For each day, it runs Monte Carlo for all 9 hours (8am–4pm), then
finds the minimum (best) and maximum (worst) hourly wait.

### View 2 — display_daily_forecast()

Shows the full hour-by-hour breakdown for one date:

```
08:00 (🌅 Morning)
   Wait: 28 minutes (🟢 LOW)
   Likely range (P10-P90): 22-35 min
   Queue: ~12 people (Week-3 Monday avg)
   [███████░░░░░░░░░░░░░]
   ✅ GOOD - Short wait (<25 min)

09:00 (🌅 Morning)
   Wait: 55 minutes (🔴 HIGH)
   ...
```

The visual bar: `█` per 4 minutes of wait (max 20 blocks = 80 min).
`█████████░░░░░░░░░░░` = 9 blocks × 4 min = ~36 min.

**Time-of-day labels:**
```python
if   hour < 12: period = "🌅 Morning"
elif hour < 13: period = "🍽️ Lunch"
else:           period = "🌆 Afternoon"
```

### View 3 — find_best_time()

Runs predictions for all 9 hours, then returns the single best (min wait)
and single worst (max wait):

```
✅ BEST TIME TO VISIT:
   🕐 08:00
   ⏱️ Wait: 28 minutes
   📉 Range (P10-P90): 22-35 min
   👥 Expected queue: ~12 people

⚠️ WORST TIME TO AVOID:
   🕐 10:00
   ⏱️ Wait: 68 minutes
   ...
```

```python
best  = min(predictions, key=lambda x: x[1])  # sort by wait, pick lowest
worst = max(predictions, key=lambda x: x[1])  # sort by wait, pick highest
```

### parse_date_input()

Validates every date the user enters:

```python
# Accepts:
"today"       → pd.Timestamp.now().normalize()
"2026-12-25"  → pd.to_datetime("2026-12-25")

# Rejects:
Sunday        → "❌ Sunday is not a working day (Mon–Sat only)."
bad format    → "❌ Invalid format. Use YYYY-MM-DD or 'today'."
```

Note: Holidays are NOT rejected here — they're allowed through and
handled inside each forecast function with the holiday guard.
This way the user gets a clear "CLOSED" message instead of a generic error.

---

## THE ENTRY POINT — predict.py

`predict.py` is the file `main.py` actually runs. Its only job is to
set up `sys.path` so Python can find all the Prediction and Preprocessing
modules, then call `cli.main()`.

Without the `sys.path` patch, running `predict.py` directly from the
project root would fail with `ModuleNotFoundError` because Python
doesn't automatically know where `src/` is.

---

## HOW A PREDICTION FLOWS FROM CLICK TO OUTPUT

When a user selects "View weekly forecast":

```
User enters date
    ↓
parse_date_input()  validates Mon–Sat only
    ↓
display_weekly_forecast(target_date)
    ↓
For each of 6 days:
    get_holiday_flags(day_date)  → if holiday: print CLOSED, skip
    ↓
    For each of 9 hours (8–16):
        predict_wait_time_monte_carlo(day_date, hour)
            ↓
            get_month_features()     → month, month_sin, month_cos, is_end_of_month
            get_holiday_flags()      → is_holiday, is_pre_holiday
            get_actual_queue_length()→ queue_length (from pattern table)
            get_actual_lag_features()→ lag_queue, lag_wait (from pattern table)
            ↓
            Generate 1,000 random samples around base values
            ↓
            model.predict(X_1000_rows)  → 1,000 wait predictions
            np.clip(wait_samples, 5, 90)
            ↓
            Return: mean, p10, p50, p90, queue_mean
    ↓
    avg_wait = mean of 9 hourly means
    best_hour = argmin(hourly_waits)
    worst_hour = argmax(hourly_waits)
    get_congestion_level(avg_wait) → level emoji + recommendation
    ↓
    Print the day's summary
```

Total predictions per weekly forecast: 6 days × 9 hours × 1,000 runs = **54,000 model.predict() calls** in one request.

---

## CONNECTION TO OTHER GUIDE FILES

| What prediction needs | Where it comes from |
|---|---|
| `queue_model.pkl` | Model Training (see Model_Training.md) — trained by `train_model.py` |
| `FEATURES` list (16 columns, exact order) | Preprocessing (see Preprocessing.md) — defined in `features.py` |
| `load_ph_holiday_month_days()` | Preprocessing (see Preprocessing.md) — `calendar.py` |
| `is_holiday` feature in model input | Preprocessing (see Preprocessing.md) — originally set by `loader.py` |
| Historical wait/queue patterns | Training data from Data Generation (see Data_Generation.md) — `synthetic_lto_cdo_queue_90days.csv` |

The prediction system is completely downstream — it only consumes outputs
from every other stage. If you change the 16 features in Preprocessing,
the model will fail to load correctly at prediction time (column mismatch).

---

## SUMMARY — What each file does

| File | What it provides |
|---|---|
| `constants.py` | `MONTE_CARLO_RUNS=1000`, seeded `RNG` generator |
| `context.py` | Loads model + patterns + holidays at startup (runs once) |
| `patterns.py` | 4-level historical lookup tables for queue/wait averages |
| `inference.py` | Feature assembly, 1,000-run Monte Carlo, congestion classifier |
| `cli.py` | 3 forecast views, holiday guard, date validator, menu loop |
| `predict.py` | Entry point: patches sys.path, calls `cli.main()` |

**The pattern across all of them:**
`context.py` does the slow work once (model load, pattern build).
`patterns.py` answers "what does history say about this slot?"
`inference.py` answers "what does the model say about this slot?"
`cli.py` answers "what should the user see?" — and blocks on holidays.

---

*End of study script — Prediction | iQueue user-facing forecast system*
