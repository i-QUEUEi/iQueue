# i-QUEUE - Project Documentation & Code Walkthrough

Detailed line-by-line code references (including loops and print statements) are in [docs/CODE_REFERENCE.md](CODE_REFERENCE.md).

This document explains the full project codebase from data generation to prediction and evaluation. It lists each file, the functions (in source order), what they do, how they interact, where outputs are written, and which other files consume those outputs.

Scope: covers files used in standard runs (data generation -> training -> prediction). Use this as a reference for maintainers, reviewers, or new contributors.

---

**Project Flow (High level):**

- Data generation: `data/Data_.py` -> generates `data/synthetic_lto_cdo_queue_90days.csv`
- Preprocessing: `src/Preprocessing/preprocess.py` -> loads CSV, engineers features, returns `X, y, features`
- Model catalog: `src/model_implementation/model_zoo/*` and `src/model_implementation/__init__.py` -> builders for candidate models
- Training & evaluation: `src/model_implementation/train_model.py` -> trains models, evaluates robustness, saves best model and outputs
- Prediction CLI: `src/Prediction/predict.py` -> loads saved model + data patterns, provides monte-carlo forecasts and console UI
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
- src/Preprocessing/preprocess.py and src/model_implementation/train_model.py load this CSV to train models and compute features.

Notes:
- This script is deterministic (seeded) and builds realistic patterns; it intentionally excludes Sundays.

---

File: src/Preprocessing/preprocess.py
- Purpose: Compatibility wrapper that re-exports the refactored preprocessing helpers.
- Why this matters: Keeps existing imports stable while the preprocessing logic is split into smaller modules.

File: src/Preprocessing/calendar.py
- load_ph_holiday_month_days(calendar_path)
  - Purpose: parse the holiday calendar into (month, day) tuples for fast membership checks.

File: src/Preprocessing/loader.py
- load_data(path)
  - Purpose: read CSV at path and apply parsing & feature engineering.
  - Steps:
    - Parse date as datetime.
    - Add month, month_sin, month_cos cyclic encodings.
    - Derive is_end_of_month (last 2 days heuristic).
    - Use calendar helper to set is_holiday and is_pre_holiday flags.
    - Compute week_of_month (1-5) via integer math on day of month.
    - Filter out negative waiting_time_min and queue_length_at_arrival rows.
    - Drop NAs and print dataset diagnostics.
  - Returns: df (cleaned pandas DataFrame).

File: src/Preprocessing/features.py
- FEATURES
  - Canonical feature order shared by training and prediction.
- get_features(df)
  - Returns: (X, y, features) where y is waiting_time_min.
- build_feature_dataframe(records, holiday_calendar_path=None)
  - Purpose: build a one-row or multi-row feature DataFrame for inference inputs.

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
This is the training orchestrator; major responsibilities are now split into focused modules:

- src/model_implementation/metrics.py
  - compute_metrics(y_true, y_pred)
  - Shared metric helper for MAE/RMSE/R2.

- src/model_implementation/evaluation.py
  - evaluate_data_quality(raw_df) for input checks.
  - evaluate_model(...) for train/test/chrono/CV evaluation and feature importance.

- src/model_implementation/splits.py
  - chronological_split(df, features) for time-aware splits.

- src/model_implementation/plots.py
  - plot_target_distribution, plot_day_hour_heatmap, plot_model_comparison, plot_actual_vs_predicted.

- src/model_implementation/reporting.py
  - write_report(...) to generate outputs/metrics.txt.

- src/model_implementation/samples.py
  - sample_predictions(model, features) sanity checks.

The train_model.py entrypoint now wires these modules together to train, select, and save the best model.

Outputs (written by this script):
- models/queue_model.pkl - serialized selected model (used by src/Prediction/predict.py).
- outputs/metrics.txt - human-readable metrics & feature importance report.
- outputs/model_comparison.csv - table used for plotting/comparison.
- outputs/plots/* - PNG visualizations.

Consumers of outputs:
- src/Prediction/predict.py loads models/queue_model.pkl and reads data/synthetic_lto_cdo_queue_90days.csv for pattern baselines.

Notes on evaluation design:
- The script computes random-split, chronological split, and CV to produce a robust_mae metric that mixes IID and time-aware evaluation.

---

File: src/Prediction/predict.py
This is the CLI entrypoint that re-exports the refactored prediction modules.

Core prediction modules:
- src/Prediction/context.py
  - Loads the trained model and historical data.
  - Builds date-aware pattern maps and prints quick sanity checks.
- src/Prediction/patterns.py
  - build_pattern_maps(...) and get_pattern_value(...) for multi-level lookup fallbacks.
- src/Prediction/inference.py
  - predict_wait_time(...) and predict_wait_time_monte_carlo(...)
  - get_congestion_level(...) plus date/holiday feature helpers.
- src/Prediction/cli.py
  - display_weekly_forecast(...), display_daily_forecast(...), find_best_time(...)
  - parse_date_input(...) and main() menu loop.

Outputs:
- Prints to console (no file outputs). Uses models/queue_model.pkl to produce predictions.

Consumers/Users:
- CLI users: end-users running python src/Prediction/predict.py.
- Could be invoked by higher-level services wrapping the CLI for integration.

---

File: main.py
- Simple orchestration script that runs training then prediction:
  - Calls python src/model_implementation/train_model.py via subprocess.run.
  - Then calls python src/Prediction/predict.py to enter CLI.

Notes:
- This is intended as a quick-run convenience; in production you might separate training (batch) and prediction (service) workflows.

---

Other repository files

- README.md - usage instructions and quick start (already provided).
- requirements.txt - runtime dependencies (un-pinned): simpy, numpy, pandas, scikit-learn, matplotlib, joblib.

---

Data & Artifact Map (where outputs are written / read)

- data/synthetic_lto_cdo_queue_90days.csv - generated by data/Data_.py; read by src/Preprocessing/loader.py and src/Prediction/context.py.
- models/queue_model.pkl - saved by src/model_implementation/train_model.py; loaded by src/Prediction/context.py.
- outputs/metrics.txt - written by train_model.py; human-facing summary.
- outputs/model_comparison.csv - written by train_model.py; machine-readable table used for plotting.
- outputs/plots/*.png - several visuals written by train_model.py (target distribution, heatmap, comparison, actual_vs_predicted).

---

How modules interact (runtime sequence)

1. (Optional) Run `python data/Data_.py` to regenerate dataset.
2. Run training: `python src/model_implementation/train_model.py` OR `python main.py` which triggers the same script.
  - `train_model.py` calls `src/Preprocessing/loader.load_data()` -> `src/Preprocessing/features.get_features()` -> builds models via `model_zoo.build_model_catalog()` -> evaluates models -> selects and joblib.dump the selected model to `models/queue_model.pkl` -> writes `outputs/*` artifacts.
3. Run prediction: `python src/Prediction/predict.py` (or let `main.py` launch it). `context.py` loads the model and data patterns, and `cli.py` renders console forecasts.

---

Practical notes & recommended improvements

- Pin `requirements.txt` versions for reproducibility (e.g., numpy==1.26.0, pandas==2.x, scikit-learn==1.x).
- Add unit tests for `preprocess.load_data`, `train_model.chronological_split`, `predict.predict_wait_time_monte_carlo` with small Monte Carlo runs.
- Add a small smoke test in CI that runs training with a smaller sample or fewer trees to ensure end-to-end compatibility.
- Consider reducing the CI Monte Carlo runs (e.g., 50) to keep test time short.
- Add graceful checks in `src/Prediction/predict.py` for missing `models/queue_model.pkl` with friendly instructions.

---

If you'd like, I can:

- Add this file to the repo (done) and open a PR with dependency pinning and a CI smoke test.
- Generate unit tests for `src/Preprocessing/preprocess.py` and `src/Prediction/predict.py`.
- Create a short `CONTRIBUTING.md` with run/test steps.

---

End of documentation.

