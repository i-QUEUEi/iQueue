# i-QUEUE - Project Documentation & Code Walkthrough

Detailed line-by-line code references (including loops and print statements) are in [docs/CODE_REFERENCE.md](CODE_REFERENCE.md).

This document explains the full project codebase from data generation to prediction and evaluation. It lists each file, the functions (in source order), what they do, how they interact, where outputs are written, and which other files consume those outputs.

Scope: covers files used in standard runs (data generation -> training -> prediction). Use this as a reference for maintainers, reviewers, or new contributors.

---

**Project Flow (High level):**

- Data generation: `data/Data_.py` -> generates `data/synthetic_lto_cdo_queue_90days.csv`
- Preprocessing: `src/preprocess.py` -> loads CSV, engineers features, returns `X, y, features`
- Model catalog: `src/model_implementation/model_zoo/*` and `src/model_implementation/__init__.py` -> builders for candidate models
- Training & evaluation: `src/model_implementation/train_model.py` -> trains models, evaluates robustness, saves best model and outputs
- Prediction CLI: `src/predict.py` -> loads saved model + data patterns, provides monte-carlo forecasts and console UI
- Orchestration: `main.py` -> convenience entrypoint that runs training then prediction

---

**Files and functions (in-sequence)**

File: data/Data_.py
- Module-level configuration/constants:
  - NUM_WEEKS, START_DATE, DATA_DIR, HOLIDAY_CALENDAR_PATH, OUTPUT_CSV_PATH, TRUE_PATTERNS - control data generation period and base hourly patterns per weekday.
- [load_ph_holidays](data/Data_.py#L31) (calendar_path, year)
  - Purpose: parse a human-readable Philippine holidays file (CSV-like text) and return a set of date objects for the given year.
  - Inputs: calendar_path (Path), year (int).
  - Output: set(datetime.date) used to apply holiday and pre-holiday modifiers during generation.
  - Consumers: internal generation loop below; this module writes holiday-aware synthetic records.

- Generation script (module-level procedural code, executed when run):
  - Iterates NUM_WEEKS starting at START_DATE.
  - For Monday->Saturday, applies seasonal factors, end-of-month, holiday and pre-holiday factors, trend and wave components to compute an hourly base_wait based on TRUE_PATTERNS (one pattern per weekday for hours 8-16).
  - For each hour, simulates num_transactions arrivals, injects noise, computes waiting_time_min, queue_length_at_arrival, service_time_min, and total_time_in_system_min.
  - Builds a pandas.DataFrame df from generated rows.
  - Creates lag features: queue_length_lag1, waiting_time_lag1 using groupby('date').shift(1) and fills first-row defaults per day.
  - Adds is_weekend, fills NAs, and saves CSV to OUTPUT_CSV_PATH (data/synthetic_lto_cdo_queue_90days.csv).

Outputs:
- data/synthetic_lto_cdo_queue_90days.csv - the core synthetic dataset used by training and prediction.

Who uses it:
- src/preprocess.py and src/model_implementation/train_model.py load this CSV to train models and compute features.

Notes:
- This script is deterministic (seeded) and builds realistic patterns; it intentionally excludes Sundays.

---

File: src/preprocess.py
- HOLIDAY_CALENDAR_PATH - resolved path to same holiday calendar file used for generation.

- [load_ph_holiday_month_days](src/preprocess.py#L9) (calendar_path)
  - Purpose: parse the same holiday calendar but return (month, day) tuples rather than date objects.
  - Output: set((month, day)) for quick membership tests used in feature engineering.

- [load_data](src/preprocess.py#L30) (path)
  - Purpose: read CSV at path and apply parsing & feature engineering.
  - Steps:
    - Parse date as datetime.
    - Add month, month_sin, month_cos cyclic encodings.
    - Derive is_end_of_month (last 2 days heuristic).
    - Use load_ph_holiday_month_days() to set is_holiday and is_pre_holiday flags.
    - Compute week_of_month (1-5) via integer math on day of month.
    - Filter out negative waiting_time_min and queue_length_at_arrival rows.
    - Drop NAs and print dataset diagnostics.
  - Returns: df (cleaned pandas DataFrame).
  - Consumers: get_features() and train_model.py.

- [get_features](src/preprocess.py#L72) (df)
  - Purpose: extract the feature matrix X and target y in the correct column order expected by the model training and predict code.
  - Feature order (important - must match training/predict code):
    - hour, day_of_week, week_of_month, month, month_sin, month_cos, is_end_of_month, is_holiday, is_pre_holiday, is_peak_day, queue_length_at_arrival, service_time_min, is_weekend, is_peak_hour, queue_length_lag1, waiting_time_lag1
  - Returns: (X, y, features) where y is waiting_time_min and features is the list above.
  - Consumers: src/model_implementation/train_model.py.

Outputs:
- No files saved by this module; it returns in-memory structures used by train_model.py.

---

File: src/model_implementation/model_zoo/__init__.py
- [build_model_catalog](src/model_implementation/model_zoo/__init__.py#L6) (random_state)
  - Purpose: assemble a dictionary of model name -> estimator ready for training.
  - Uses builders from sibling modules (RandomForest, GradientBoosting, LinearRegression).
  - Returns: {'LinearRegression': LinearRegression(), 'RandomForest': RandomForest(...), 'GradientBoosting': GradientBoosting(...)}
  - Consumers: src/model_implementation/train_model.py.

File: src/model_implementation/model_zoo/random_forest.py
- [build_random_forest](src/model_implementation/model_zoo/random_forest.py#L4) (random_state, params=None)
  - Returns a RandomForestRegressor with default hyperparameters (500 trees, max_depth 15, etc.).
  - params can override defaults.

File: src/model_implementation/model_zoo/gradient_boosting.py
- [build_gradient_boosting](src/model_implementation/model_zoo/gradient_boosting.py#L4) (random_state, params=None)
  - Returns GradientBoostingRegressor with tuned defaults (250 trees, lr=0.05, depth=3).

File: src/model_implementation/model_zoo/linear_regression.py
- [build_linear_regression](src/model_implementation/model_zoo/linear_regression.py#L4) ()
  - Returns LinearRegression() (baseline model).

Notes:
- These builder functions are small but central: they centralize hyperparameter choices so train_model.py can benchmark multiple model classes with consistent seeds.

---

File: src/model_implementation/train_model.py
This is the core training & evaluation script; order of functions as implemented:

- [compute_metrics](src/model_implementation/train_model.py#L34) (y_true, y_pred)
  - Returns mae, rmse, and r2 as a dictionary.

- [ensure_output_dirs](src/model_implementation/train_model.py#L42) ()
  - Creates outputs/ and outputs/plots/ if missing.

- [evaluate_data_quality](src/model_implementation/train_model.py#L47) (raw_df)
  - Runs quick data checks: duplicates, missing cells, negative values, and target distribution summaries.
  - Returns a summary dict used in the metrics.txt report.

- [plot_target_distribution](src/model_implementation/train_model.py#L68) (df)
  - Saves outputs/plots/target_distribution.png showing histogram of waiting_time_min.

- [plot_day_hour_heatmap](src/model_implementation/train_model.py#L79) (df)
  - Saves outputs/plots/day_hour_heatmap.png showing average waiting time by day/hour matrix.

- [plot_model_comparison](src/model_implementation/train_model.py#L100) (results_df)
  - Saves outputs/plots/model_comparison.png comparing MAE across models and evaluation splits.

- [plot_actual_vs_predicted](src/model_implementation/train_model.py#L119) (y_true, y_pred, model_name)
  - Saves outputs/plots/actual_vs_predicted.png scatter plot of actual vs predicted.

- [get_feature_importance](src/model_implementation/train_model.py#L132) (model, features, X_reference, y_reference)
  - Returns a DataFrame of normalized feature importances using tree importances, coefficients, or permutation importance fallback.

- [evaluate_model](src/model_implementation/train_model.py#L155) (name, model, X_train, X_test, y_train, y_test, chrono_train, chrono_test, features)
  - Fits the model on X_train and computes: train metrics, test metrics, chronological test metrics (train on earlier dates, test on later), cross-validated metrics, robust MAE (mean of test, chrono, CV MAE), p90/p95 absolute errors, feature importance, and segment error tables.
  - Returns a large dict with fitted model and diagnostics.

- [chronological_split](src/model_implementation/train_model.py#L243) (df, features)
  - Produces a time-based training/test split at 80% of unique dates (preserves chronology). Returns (X_time_train, y_time_train), (X_time_test, y_time_test), time_train_dates_count, time_test_dates_count.

- [sample_predictions](src/model_implementation/train_model.py#L265) (model, features)
  - Emits example predictions using fixed cases printed to stdout for sanity checks.

- [write_report](src/model_implementation/train_model.py#L304) (summary, results_df, selected_result, baseline_metrics, time_train_dates, time_test_dates)
  - Writes outputs/metrics.txt with the data evaluation, per-model metrics, baseline comparisons, and feature importances.

- [main](src/model_implementation/train_model.py#L380) ()
  - End-to-end orchestration for training:
    1. Ensure output directories.
    2. Read raw CSV into raw_df and get summary via evaluate_data_quality.
    3. Use load_data and get_features from src/preprocess.py to prepare X, y, features.
    4. Train/test random split with train_test_split and chronological split with chronological_split.
    5. Build model catalog (build_model_catalog) and evaluate each candidate via evaluate_model.
    6. Select model by minimal robust_mae, save it to models/queue_model.pkl (via joblib.dump).
    7. Save plots to outputs/plots, save outputs/model_comparison.csv, and write outputs/metrics.txt.

Outputs (written by this script):
- models/queue_model.pkl - serialized selected model (used by src/predict.py).
- outputs/metrics.txt - human-readable metrics & feature importance report.
- outputs/model_comparison.csv - table used for plotting/comparison.
- outputs/plots/* - PNG visualizations.

Consumers of outputs:
- src/predict.py loads models/queue_model.pkl and reads data/synthetic_lto_cdo_queue_90days.csv for pattern baselines.

Notes on evaluation design:
- The script computes random-split, chronological split, and CV to produce a robust_mae metric that mixes IID and time-aware evaluation.

---

File: src/predict.py
This module is the CLI used to produce human-friendly forecasts and Monte Carlo uncertainty.

Top-level module behavior:
- Resolves BASE_DIR, sets MODEL_PATH (models/queue_model.pkl) and DATA_PATH (data/synthetic_lto_cdo_queue_90days.csv).
- Loads the saved model = joblib.load(MODEL_PATH) on import.
- Loads the CSV data as df and computes week_of_month and month for pattern extraction.

Functions (in file order):
- [load_ph_holiday_month_days](src/predict.py#L32) (calendar_path)
  - Parses holiday calendar into (month, day) tuples similar to preprocess.py.

- [build_pattern_maps](src/predict.py#L55) (value_col)
  - Scans df and builds nested dicts of averages keyed by day_name, week_of_month, month, and hour for both queue_length_at_arrival and waiting_time_min.
  - These maps are used as an empirical baseline/seed when forming ML features at prediction time.

- [get_pattern_value](src/predict.py#L100) (pattern_maps, day_name, month, week, hour, default_value)
  - Lookup function that falls back through four levels of aggregation: day+month+week+hour -> day+month+hour -> day+week+hour -> day+hour -> default.

- [get_actual_queue_length](src/predict.py#L150) (day_name, month, week, hour)
  - Convenience wrapper returning queue baseline for a requested slot.

- [get_actual_lag_features](src/predict.py#L154) (day_name, month, week, hour, queue_length)
  - Returns lag1 queue + wait features using previous hour if available, otherwise uses defaults.

- [get_month_features](src/predict.py#L165) (target_date)
  - Computes cyclic month features (month_sin, month_cos) and is_end_of_month.

- [get_holiday_flags](src/predict.py#L174) (target_date)
  - Returns is_holiday and is_pre_holiday using parsed calendar.

- [predict_wait_time](src/predict.py#L182) (day_name, target_date, hour)
  - Builds the single-row feature vector matching FEATURES order and calls model.predict(X).

- [predict_wait_time_monte_carlo](src/predict.py#L226) (target_date, hour, runs=1000)
  - Produces Monte Carlo samples by perturbing operational features (queue, service time, lag features) using Gaussian noise and clipping. Uses model.predict(X_samples) to compute distribution (mean, p10, p50, p90) and queue_mean.
  - Note: MONTE_CARLO_RUNS default = 1000 (may be heavy for CI / quick runs).

- [get_congestion_level](src/predict.py#L291) (wait_time)
  - Maps wait time to three levels (LOW/MODERATE/HIGH) and returns user-facing recommendation text.

- [display_weekly_forecast](src/predict.py#L300) (target_date)
  - Iterates the Monday-Saturday of the target week and prints per-day summaries using Monte Carlo results per hour.

- [display_daily_forecast](src/predict.py#L328) (target_date)
  - Prints hourly breakdown for a single date with bar visualization and P10-P90 ranges.

- [find_best_time](src/predict.py#L358) (target_date)
  - Finds the hour with lowest mean wait and prints the best/worst times for visiting.

- [parse_date_input](src/predict.py#L392) (prompt)
  - Simple console prompt utility; accepts YYYY-MM-DD or today and enforces Mon-Sat only.

- [main](src/predict.py#L410) ()
  - Console loop with options to view weekly forecast, daily forecast, best time, or exit. Uses above functions and Monte Carlo predictions.

Outputs:
- Prints to console (no file outputs). Uses models/queue_model.pkl to produce predictions.

Consumers/Users:
- CLI users: end-users running python src/predict.py.
- Could be invoked by higher-level services wrapping the CLI for integration.

---

File: main.py
- Simple orchestration script that runs training then prediction:
  - Calls python src/model_implementation/train_model.py via subprocess.run.
  - Then calls python src/predict.py to enter CLI.

Notes:
- This is intended as a quick-run convenience; in production you might separate training (batch) and prediction (service) workflows.

---

Other repository files

- README.md - usage instructions and quick start (already provided).
- requirements.txt - runtime dependencies (un-pinned): simpy, numpy, pandas, scikit-learn, matplotlib, joblib.

---

Data & Artifact Map (where outputs are written / read)

- data/synthetic_lto_cdo_queue_90days.csv - generated by data/Data_.py; read by src/preprocess.py and src/predict.py.
- models/queue_model.pkl - saved by src/model_implementation/train_model.py; loaded by src/predict.py.
- outputs/metrics.txt - written by train_model.py; human-facing summary.
- outputs/model_comparison.csv - written by train_model.py; machine-readable table used for plotting.
- outputs/plots/*.png - several visuals written by train_model.py (target distribution, heatmap, comparison, actual_vs_predicted).

---

How modules interact (runtime sequence)

1. (Optional) Run `python data/Data_.py` to regenerate dataset.
2. Run training: `python src/model_implementation/train_model.py` OR `python main.py` which triggers the same script.
   - `train_model.py` calls `src/preprocess.load_data()` -> `get_features()` -> builds models via `model_zoo.build_model_catalog()` -> evaluates models -> selects and joblib.dump the selected model to `models/queue_model.pkl` -> writes `outputs/*` artifacts.
3. Run prediction: `python src/predict.py` (or let `main.py` launch it). `predict.py` loads the model and data patterns, and returns console forecasts.

---

Practical notes & recommended improvements

- Pin `requirements.txt` versions for reproducibility (e.g., numpy==1.26.0, pandas==2.x, scikit-learn==1.x).
- Add unit tests for `preprocess.load_data`, `train_model.chronological_split`, `predict.predict_wait_time_monte_carlo` with small Monte Carlo runs.
- Add a small smoke test in CI that runs training with a smaller sample or fewer trees to ensure end-to-end compatibility.
- Consider reducing the CI Monte Carlo runs (e.g., 50) to keep test time short.
- Add graceful checks in `src/predict.py` for missing `models/queue_model.pkl` with friendly instructions.

---

If you'd like, I can:

- Add this file to the repo (done) and open a PR with dependency pinning and a CI smoke test.
- Generate unit tests for `src/preprocess.py` and `src/predict.py`.
- Create a short `CONTRIBUTING.md` with run/test steps.

---

End of documentation.
