# iQueue — How We Built This From Scratch

> This document tells the full story of building iQueue: from the problem we faced, to generating data, cleaning it, training models, evaluating them, and finally making predictions. Every file and folder is mentioned in the order we actually worked on it.

---

## The Problem

We wanted to answer one question: **"How long will I wait at the LTO CDO office?"**

Queue waiting times at government offices like LTO (Land Transportation Office) are unpredictable. They depend on the day of the week, the time of day, whether it's near a holiday, which week of the month it is, and how many people are already in line. A simple guess ("maybe 30 minutes?") isn't useful. We needed a machine learning model that could give real, data-driven predictions.

---

## Chapter 1: We Had No Real Data — So We Built Our Own

**Folder:** [`data/`](../data/)
**File:** [`data/Data_.py`](../data/Data_.py)
**Output:** [`data/synthetic_lto_cdo_queue_90days.csv`](../data/synthetic_lto_cdo_queue_90days.csv)

The first challenge: we had no real LTO queue logs to train on. So we did what data scientists often do early in a project — we **simulated realistic data**.

We wrote `Data_.py` to generate 90 days of synthetic queue transactions. It's not random noise — every row is carefully constructed to reflect real patterns:

- **Mondays and Fridays are busier.** The `TRUE_PATTERNS` dictionary defines different hourly waiting-time targets for each weekday. Monday 9am has a much higher base wait than Wednesday 9am.
- **9am–11am is peak time.** The hour loop (hours 8–16) applies hourly multipliers based on the true pattern for that day.
- **Holidays and pre-holidays spike.** The `load_ph_holidays()` function reads [`data/2026-calendar-with-holidays-portrait-sunday-start-en-ph.csv`](../data/2026-calendar-with-holidays-portrait-sunday-start-en-ph.csv) and returns a set of dates. When the generation loop lands on a holiday, it applies a multiplier to push wait times up.
- **End-of-month is busier.** If the date is in the last 3 days of the month, another multiplier is applied — because more people rush to renew licenses before the month ends.
- **Noise is added.** Each simulated transaction gets small random variation so the model doesn't see perfect patterns and overfit.

After generating all rows, we compute **lag features**:
- `queue_length_lag1` — the queue length from the previous transaction that day
- `waiting_time_lag1` — the waiting time from the previous transaction that day

These are important because the queue has **momentum** — if the previous person waited 60 minutes, you probably will too.

Finally, `is_weekend` is added, NaNs are dropped, and the CSV is saved. This one file — `synthetic_lto_cdo_queue_90days.csv` — is the backbone of the entire project. Everything downstream depends on it.

---

## Chapter 2: Cleaning the Data and Extracting Features

**Folder:** [`src/Preprocessing/`](../src/Preprocessing/)
**Files:**
- [`src/Preprocessing/calendar.py`](../src/Preprocessing/calendar.py) — holiday parser
- [`src/Preprocessing/loader.py`](../src/Preprocessing/loader.py) — loads and cleans the CSV
- [`src/Preprocessing/features.py`](../src/Preprocessing/features.py) — defines and extracts model features
- [`src/Preprocessing/preprocess.py`](../src/Preprocessing/preprocess.py) — single import that bundles all of the above

### Step 1: Parsing holidays

[`calendar.py`](../src/Preprocessing/calendar.py) contains one function: `load_ph_holiday_month_days()`. It reads the Philippine holiday calendar file and extracts `(month, day)` tuples using a regex pattern. The result is a Python `set` so membership checks (like "is January 1 a holiday?") are fast.

### Step 2: Loading and cleaning

[`loader.py`](../src/Preprocessing/loader.py) — `load_data()` does the heavy lifting:

1. Reads the CSV into a DataFrame
2. Parses the `date` column as a proper datetime object
3. Extracts `month` as a number
4. Encodes month **cyclically** using `month_sin` and `month_cos` — this is important because December (12) and January (1) are close in real life but far apart numerically. The sin/cos trick wraps them together on a circle.
5. Flags `is_end_of_month` if the day is within 3 days of the end of the month
6. Calls `calendar.py` to flag `is_holiday` and `is_pre_holiday` for each row
7. Derives `week_of_month` (1–4) using integer division of the day number
8. **Removes bad rows:** drops rows where `waiting_time_min < 0` or `queue_length_at_arrival < 0` — those are impossible in real life
9. Drops any remaining NaN rows
10. Prints a summary so we can verify the data looks right

### Step 3: Defining exactly what the model can see

[`features.py`](../src/Preprocessing/features.py) contains the `FEATURES` list — the 16 columns the model is trained on:

```
hour, day_of_week, week_of_month, month, month_sin, month_cos,
is_end_of_month, is_holiday, is_pre_holiday, is_peak_day,
queue_length_at_arrival, service_time_min, is_weekend, is_peak_hour,
queue_length_lag1, waiting_time_lag1
```

`get_features(df)` slices the DataFrame to just these 16 columns (X) and returns `waiting_time_min` as the target (y). This exact same list is reused at prediction time — which is critical because the model must see the same columns in the same order every time.

`build_feature_dataframe()` is used during prediction to convert a query (e.g. "Monday, 9am, 25 people in queue") into a properly-formatted row for the model.

### Step 4: The wrapper

[`preprocess.py`](../src/Preprocessing/preprocess.py) is just 11 lines. It re-exports `load_data`, `get_features`, `build_feature_dataframe`, and `FEATURES` from the other three files, so any other script can do `from Preprocessing.preprocess import load_data` without needing to know the internal file structure.

---

## Chapter 3: Building the Models

**Folder:** [`src/model_implementation/model_zoo/`](../src/model_implementation/model_zoo/)
**Files:**
- [`linear_regression.py`](../src/model_implementation/model_zoo/linear_regression.py)
- [`random_forest.py`](../src/model_implementation/model_zoo/random_forest.py)
- [`gradient_boosting.py`](../src/model_implementation/model_zoo/gradient_boosting.py)
- [`__init__.py`](../src/model_implementation/model_zoo/__init__.py) — assembles the model menu
- [`src/model_implementation/__init__.py`](../src/model_implementation/__init__.py) — re-exports `build_model_catalog`

We didn't pick one model and hope for the best. We tested **three different model types** and let the evaluation decide which is best.

### Model 1 — Linear Regression
[`linear_regression.py`](../src/model_implementation/model_zoo/linear_regression.py) simply returns `LinearRegression()`. This is our **baseline**. It draws a straight line through all the data. If the relationship between inputs and wait time were perfectly linear ("every extra person in queue = exactly 2 more minutes"), this would be perfect. But queue dynamics are messier than that.

### Model 2 — Random Forest
[`random_forest.py`](../src/model_implementation/model_zoo/random_forest.py) builds a `RandomForestRegressor` with these settings:
- `n_estimators=500` — 500 decision trees vote together
- `max_depth=15` — each tree can go 15 levels deep
- `min_samples_leaf=2` — each leaf needs at least 2 samples (prevents overfitting)
- `max_features="sqrt"` — each tree sees only a random subset of features (adds diversity)

> **Analogy:** Like asking 500 different people the same question and taking the average answer. Each person saw slightly different data, so the combined answer is more reliable than any single estimate.

### Model 3 — Gradient Boosting
[`gradient_boosting.py`](../src/model_implementation/model_zoo/gradient_boosting.py) builds a `GradientBoostingRegressor` with:
- `n_estimators=250` — 250 trees, but each one learns from the previous tree's errors
- `learning_rate=0.05` — small steps so it doesn't overshoot
- `max_depth=3` — shallow trees to prevent overfitting
- `subsample=0.9` — uses 90% of data per tree (adds randomness)

> **Analogy:** Like a student who re-reads only the questions they got wrong. Each tree focuses on the mistakes the previous trees made.

### The model menu
[`model_zoo/__init__.py`](../src/model_implementation/model_zoo/__init__.py) — `build_model_catalog()` returns a dictionary:
```python
{
  "LinearRegression": build_linear_regression(),
  "RandomForest": build_random_forest(random_state),
  "GradientBoosting": build_gradient_boosting(random_state),
}
```
This is the central "menu" of candidates. The training script loops through all of them.

---

## Chapter 4: Training and Evaluating Every Model

**File:** [`src/model_implementation/train_model.py`](../src/model_implementation/train_model.py)

This is the heart of the project. It orchestrates the entire training pipeline:

1. Creates `outputs/` and `outputs/plots/` directories
2. Loads the raw CSV and runs `evaluate_data_quality()` on it (before any cleaning)
3. Calls `load_data()` to clean the data and add features
4. Calls `get_features()` to extract X and y
5. Splits into train/test sets (80/20 random split)
6. Gets the chronological split from [`splits.py`](../src/Evaluation/splits.py)
7. Loads the model catalog and loops through each model

### The Evaluation Folder — `src/Evaluation/`

**Folder:** [`src/Evaluation/`](../src/Evaluation/)

This folder holds all evaluation logic, completely separate from the training script. It was designed this way so evaluation can be updated without touching training code.

#### `evaluation.py` — the core evaluator
[`src/Evaluation/evaluation.py`](../src/Evaluation/evaluation.py)

`evaluate_model()` runs each model through **three different tests**:

**Test 1 — Random split** ([L45–49](../src/Evaluation/evaluation.py#L45-L49))
Trains on the 80% random train set, predicts on the 20% test set. This is the standard machine learning test.

**Test 2 — Chronological split** ([L51–53](../src/Evaluation/evaluation.py#L51-L53))
A completely separate model is cloned and trained on the oldest 80% of dates, tested on the newest 20%. This tests whether the model generalizes to future dates — much more realistic than a random split for time-based data.

**Test 3 — 5-fold cross-validation** ([L55–66](../src/Evaluation/evaluation.py#L55-L66))
Splits the training data into 5 equal chunks. Trains on 4, tests on 1. Repeats 5 times. This gives a stable average score across many different train/test combinations.

All three MAE scores are averaged into **`robust_mae`** ([L75](../src/Evaluation/evaluation.py#L75)):
```python
robust_mae = float(np.mean([test_metrics["mae"], chrono_metrics["mae"], cv_mae]))
```
The model with the **lowest `robust_mae`** is selected as the winner.

The function also computes **error percentiles** ([L77–80](../src/Evaluation/evaluation.py#L77-L80)) — P90, P95, and max absolute error — so we know the worst-case behavior, not just the average.

It also segments errors by day and hour ([L97–105](../src/Evaluation/evaluation.py#L97-L105)) to check if the model is significantly worse on peak days vs normal days.

`evaluate_data_quality()` ([L137–155](../src/Evaluation/evaluation.py#L137-L155)) checks the **raw** CSV (before any cleaning) for duplicates, missing values, negative waits, and prints target distribution statistics.

#### `metrics.py` — shared error calculator
[`src/Evaluation/metrics.py`](../src/Evaluation/metrics.py)

One function: `compute_metrics(y_true, y_pred)` that returns a dictionary of MAE, RMSE, and R². Used everywhere metrics need to be calculated.

#### `splits.py` — time-aware split
[`src/Evaluation/splits.py`](../src/Evaluation/splits.py)

`chronological_split()` sorts all data by date, finds all unique dates, cuts at 80%, and returns the oldest 80% as training and newest 20% as test. This is much harder to "cheat" than a random split because the model has never seen any of the test dates during training.

#### `plots.py` — 4 charts saved to disk
[`src/Evaluation/plots.py`](../src/Evaluation/plots.py)

After training, four PNG charts are generated and saved to [`outputs/plots/`](../outputs/plots/):

| Chart | File | What it shows |
|---|---|---|
| `plot_target_distribution()` | `target_distribution.png` | Histogram of all waiting times — are they mostly short or long? |
| `plot_day_hour_heatmap()` | `day_hour_heatmap.png` | Color grid: which day+hour combinations are busiest? |
| `plot_model_comparison()` | `model_comparison.png` | Bar chart comparing all 3 models' MAE scores side by side |
| `plot_actual_vs_predicted()` | `actual_vs_predicted.png` | Scatter plot: actual vs predicted wait times. Points near the diagonal = good predictions |

#### `reporting.py` — the full written report
[`src/Evaluation/reporting.py`](../src/Evaluation/reporting.py)

`write_report()` generates [`outputs/metrics.txt`](../outputs/metrics.txt) — a plain text file with 8 sections explaining everything: data quality, model benchmark results, baseline comparison, robust evaluation, segment error checks, why these models were chosen, why others weren't, and preprocessing rationale.

#### `samples.py` — sanity check
[`src/Evaluation/samples.py`](../src/Evaluation/samples.py)

After saving the model, 6 hardcoded test cases are run through it. These are scenarios where we already know roughly what the answer should be:
- "Monday 9am, 25 people in queue → should be ~55 min"
- "Wednesday 8am, 4 people → should be ~9 min"

If these are wildly wrong, something broke in training.

### What training produces

After the training pipeline finishes, these outputs are saved:

| Output | Where |
|---|---|
| Trained model (best) | [`models/queue_model.pkl`](../models/queue_model.pkl) |
| Model comparison table | [`outputs/model_comparison.csv`](../outputs/model_comparison.csv) |
| Full evaluation report | [`outputs/metrics.txt`](../outputs/metrics.txt) |
| 4 visualization charts | [`outputs/plots/*.png`](../outputs/plots/) |

---

## Chapter 5: Making Predictions — The User-Facing App

**Folder:** [`src/Prediction/`](../src/Prediction/)
**Files:**
- [`constants.py`](../src/Prediction/constants.py) — shared settings
- [`context.py`](../src/Prediction/context.py) — loads model and builds pattern tables
- [`patterns.py`](../src/Prediction/patterns.py) — historical pattern lookup
- [`inference.py`](../src/Prediction/inference.py) — builds inputs and runs predictions
- [`cli.py`](../src/Prediction/cli.py) — the interactive menu
- [`predict.py`](../src/Prediction/predict.py) — entry point that sets up paths and starts the CLI

### `constants.py`
[`constants.py`](../src/Prediction/constants.py) sets two values: `MONTE_CARLO_RUNS = 1000` (how many simulation runs per prediction) and `RNG = np.random.default_rng(42)` (a seeded random number generator so results are reproducible).

### `context.py` — startup: loading everything
[`context.py`](../src/Prediction/context.py) runs once when the prediction app starts:

1. Loads the saved model from `models/queue_model.pkl` using `joblib.load()`
2. Reads the original CSV back into a DataFrame
3. Adds `week_of_month` and `month` columns
4. Calls `build_pattern_maps()` from [`patterns.py`](../src/Prediction/patterns.py) twice — once for `queue_length_at_arrival` and once for `waiting_time_min`
5. Loads the holiday calendar
6. Computes `avg_service_time` from the data

The pattern maps and model are stored as module-level variables so every part of the prediction code can import them.

### `patterns.py` — the historical lookup table
[`patterns.py`](../src/Prediction/patterns.py)

`build_pattern_maps()` builds a 4-level nested lookup table from the training data:
- Level 1: day name (Monday, Tuesday…)
- Level 2: month (1–12)
- Level 3: week of month (1–4)
- Level 4: hour (8–16)

At each leaf, it stores the **average** queue length (or wait time) from the actual data for that exact slot.

`get_pattern_value()` looks up a value using all 4 levels. If data is missing at the finest level (e.g. "Tuesday, March, Week 3, 14:00" had no records), it **falls back** to coarser levels: month+day → week+day → just day+hour → finally the default value. This ensures the app always returns something sensible.

### `inference.py` — the actual prediction
[`inference.py`](../src/Prediction/inference.py)

This is where the ML model gets used. There are two main functions:

**`predict_wait_time()`** — single deterministic prediction:
1. Calculates all 16 feature values for the requested date and hour (day of week, week of month, month sin/cos, holiday flags, queue from pattern lookup, lag features)
2. Packs them into a one-row DataFrame in the exact `FEATURES` column order
3. Calls `model.predict(X)[0]` and returns the result

**`predict_wait_time_monte_carlo()`** — 1,000 simulations:
1. Gets the same base feature values
2. Adds **random noise** to the uncertain inputs (queue size ±15%, service time ±10%, lag features ±20%)
3. Runs all 1,000 variations through the model at once
4. Returns `mean`, `p10`, `p50`, `p90` — a full probability distribution

> **Analogy:** Instead of saying "you'll wait exactly 32 minutes", it says "most likely 28–36 minutes, with 90% confidence". Like a weather forecast.

**`get_congestion_level()`** classifies the wait time:
- Over 45 min → 🔴 HIGH
- 25–45 min → 🟡 MODERATE
- Under 25 min → 🟢 LOW

### `cli.py` — the interactive menu
[`cli.py`](../src/Prediction/cli.py)

`main()` runs an infinite loop showing 4 options:

| Option | Function called | What it does |
|---|---|---|
| 1 — Weekly forecast | `display_weekly_forecast()` | Loops through all 6 working days of the week, runs Monte Carlo for each of 9 hours per day, shows best/worst times |
| 2 — Specific date | `display_daily_forecast()` | Runs Monte Carlo for all 9 hours of one chosen date, shows a bar chart and congestion level per hour |
| 3 — Best time | `find_best_time()` | Runs all 9 hours, finds the minimum mean wait, shows best and worst hours |
| 4 — Exit | breaks the loop | Goodbye message |

`parse_date_input()` handles user date input. It accepts `"today"` or `"YYYY-MM-DD"` format, rejects Sundays (the office is closed), and keeps asking until a valid date is entered.

### `predict.py` — the entry point
[`predict.py`](../src/Prediction/predict.py)

Sets up `sys.path` so Python can find all the Prediction and Preprocessing modules regardless of where the script is run from, then imports everything and calls `main()`.

---

## Chapter 6: Tying It All Together

**File:** [`main.py`](../main.py)

`main.py` is just 10 lines. It uses Python's `subprocess` module to run the two phases in order:

```python
subprocess.run([sys.executable, "src/model_implementation/train_model.py"], check=True)
subprocess.run([sys.executable, "src/Prediction/predict.py"], check=True)
```

The `check=True` flag means if training fails, prediction won't start. This prevents the app from trying to load a model file that doesn't exist yet.

---

## Full File and Folder Map

```
iQueue/
│
├── main.py                          ← Start here: runs training then prediction
│
├── data/
│   ├── Data_.py                     ← Generates the 90-day synthetic dataset
│   ├── synthetic_lto_cdo_queue_90days.csv  ← The training data (output of Data_.py)
│   └── 2026-calendar-with-holidays-...csv  ← Philippine holiday calendar
│
├── src/
│   ├── Preprocessing/
│   │   ├── calendar.py              ← Parses holiday dates from the calendar file
│   │   ├── loader.py                ← Loads CSV, adds features, cleans rows
│   │   ├── features.py              ← Defines FEATURES list + builds feature DataFrames
│   │   └── preprocess.py            ← Single import wrapper for the 3 files above
│   │
│   ├── model_implementation/
│   │   ├── __init__.py              ← Re-exports build_model_catalog
│   │   ├── train_model.py           ← Full training pipeline orchestrator
│   │   └── model_zoo/
│   │       ├── __init__.py          ← build_model_catalog(): assembles model menu
│   │       ├── linear_regression.py ← Model 1: baseline linear model
│   │       ├── random_forest.py     ← Model 2: 500-tree ensemble
│   │       └── gradient_boosting.py ← Model 3: sequential boosted trees
│   │
│   └── Evaluation/
│       ├── __init__.py              ← Makes Evaluation a Python package
│       ├── evaluation.py            ← evaluate_model() + evaluate_data_quality()
│       ├── metrics.py               ← compute_metrics(): MAE, RMSE, R²
│       ├── splits.py                ← chronological_split() for time-aware testing
│       ├── plots.py                 ← 4 chart generators (PNG output)
│       ├── reporting.py             ← write_report(): full metrics.txt writer
│       └── samples.py               ← sample_predictions(): sanity check runs
│
│   └── Prediction/
│       ├── predict.py               ← Entry point: sets up paths, calls main()
│       ├── constants.py             ← MONTE_CARLO_RUNS=1000, RNG seed
│       ├── context.py               ← Loads model + builds pattern lookup tables
│       ├── patterns.py              ← build_pattern_maps() + get_pattern_value()
│       ├── inference.py             ← predict_wait_time() + Monte Carlo simulation
│       └── cli.py                   ← Interactive menu (weekly/daily/best-time)
│
├── models/
│   └── queue_model.pkl              ← Saved best model (output of training)
│
├── outputs/
│   ├── metrics.txt                  ← Full evaluation report
│   ├── model_comparison.csv         ← Model benchmark table
│   └── plots/
│       ├── target_distribution.png
│       ├── day_hour_heatmap.png
│       ├── model_comparison.png
│       └── actual_vs_predicted.png
│
└── docs/
    ├── PROJECT_DOCUMENTATION.md     ← This file
    ├── CODE_REFERENCE.md            ← Detailed per-file, per-function, per-line reference
    ├── Evaluation_detailes.md       ← Deep dive on all evaluation methods with code links
    ├── Models_Comparison.md         ← Why RandomForest won with code evidence
    └── PRESENTATION_REQS.md         ← Checklist of requirements vs what was built
```

---

## The Full Build Sequence (Summary)

```
Step 1: data/Data_.py
   → Simulates 90 days of queue data with real patterns
   → Output: data/synthetic_lto_cdo_queue_90days.csv

Step 2: src/Preprocessing/ (called by train_model.py)
   → calendar.py  reads the holiday calendar
   → loader.py    cleans data, adds time/holiday/cyclic features
   → features.py  extracts the 16-column feature matrix X and target y

Step 3: src/model_implementation/model_zoo/ (called by train_model.py)
   → Builds LinearRegression, RandomForest, GradientBoosting

Step 4: src/Evaluation/ (called by train_model.py)
   → evaluation.py  tests each model 3 ways (random + chrono + CV)
   → metrics.py     computes MAE, RMSE, R²
   → splits.py      handles the chronological train/test split
   → Selects winner by lowest robust_mae

Step 5: src/Evaluation/ (output phase)
   → plots.py      saves 4 charts to outputs/plots/
   → reporting.py  saves full report to outputs/metrics.txt
   → samples.py    runs sanity-check predictions
   → train_model.py saves winner to models/queue_model.pkl

Step 6: src/Prediction/ (run after training)
   → context.py    loads model + builds pattern lookup tables
   → patterns.py   provides historical queue averages per day/hour
   → inference.py  runs Monte Carlo predictions (1000 simulations per hour)
   → cli.py        shows the interactive menu to the user
```
