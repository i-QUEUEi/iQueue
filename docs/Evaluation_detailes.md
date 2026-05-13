# Evaluation Details

This document explains every evaluation step used in this project, what each metric means, the results from the latest run, and whether the evaluation approach is standard. Metrics and values below come from [outputs/metrics.txt](outputs/metrics.txt) and [outputs/model_comparison.csv](outputs/model_comparison.csv).

## 1. Data Quality Evaluation (Input Sanity Checks)

**Purpose:** Confirm the dataset is clean and suitable for training. This is a standard prerequisite for any ML project.

**Checks and results:**

- Row count: 21,428
- Columns: 20
- Duplicate rows: 0
- Missing cells: 0
- Negative waiting-time rows: 0
- Negative queue rows: 0

**Target distribution (waiting_time_min):**

- Mean: 36.56
- Median: 28.60
- Standard deviation: 22.05
- Min / Max: 6.60 / 90.00
- P10 / P90: 14.90 / 73.60

**Is this standard?** Yes. Basic integrity checks (duplicates, missing, invalid values) and a summary of the target distribution are standard and necessary in every supervised learning pipeline.

## 2. Model Benchmarking (Model-to-Model Comparison)

**Purpose:** Compare multiple model families and select the best one using consistent evaluation metrics. This is standard practice for tabular ML problems.

**Models compared:**

- `RandomForest`
- `GradientBoosting`
- `LinearRegression`

**Key results (test and robust metrics):**

- RandomForest: Test MAE 2.92, Chrono MAE 2.74, CV MAE 2.89, Robust MAE 2.85, Test R2 0.9637
- GradientBoosting: Test MAE 3.09, Chrono MAE 2.93, CV MAE 3.00, Robust MAE 3.01, Test R2 0.9613
- LinearRegression: Test MAE 4.44, Chrono MAE 4.12, CV MAE 4.39, Robust MAE 4.32, Test R2 0.9248

**Why the selected model wins:** RandomForest has the lowest robust MAE and the highest R2 across splits. It provides the best balance of accuracy and stability.

**Is this standard?** Yes. Benchmarking multiple models and picking the best by a stable metric (like robust MAE) is standard.

## 3. Baseline Comparison

**Purpose:** Ensure the model is better than a trivial predictor. This prevents reporting high numbers without context.

**Baseline used:** Mean predictor (always predicts the average waiting time).

**Baseline results:**

- Test MAE: 18.18
- Test RMSE: 22.28
- Test R2: -0.0002

**Interpretation:** The baseline performs very poorly compared to the ML models. This confirms the models are learning meaningful patterns.

**Is this standard?** Yes. A baseline comparison is standard in ML evaluation.

## 4. Robust Evaluation (Multiple Splits)

**Purpose:** Make sure performance is not an accident of one split. This is important when data has time structure.

**Evaluations used:**

- Random train/test split
- Chronological split (train on earlier dates, test on later dates)
- Cross-validation (5-fold)

**Random split (RandomForest):**

- MAE: 2.92
- RMSE: 4.24
- R2: 0.9637

**Chronological split (RandomForest):**

- Train dates: 176
- Test dates: 44
- MAE: 2.74
- RMSE: 4.03
- R2: 0.9618

**Cross-validation (RandomForest):**

- CV MAE: 2.89
- CV RMSE: 4.19
- CV R2: 0.9636

**Is this standard?** Yes. Random splits and cross-validation are standard. Chronological splits are the standard approach when time effects exist. Using all three is a strong and responsible evaluation practice.

## 5. Error Distribution (Tail Risk)

**Purpose:** Understand how large errors get in worst cases, not just on average.

**Results (RandomForest):**

- P90 absolute error: 7.20
- P95 absolute error: 9.78
- Max absolute error: 25.23

**Interpretation:** Most predictions are within about 7 to 10 minutes. Rare worst-case errors can reach about 25 minutes.

**Is this standard?** Yes. Reporting percentile errors (P90, P95) is a common way to communicate tail risk.

## 6. Segment Error Checks (Peak vs Non-Peak)

**Purpose:** Confirm performance does not collapse on busy periods.

**Results (RandomForest):**

- Peak day MAE (Mon/Fri): 4.66
- Non-peak day MAE: 1.76
- Peak hour MAE: 3.70
- Non-peak hour MAE: 1.95

**Interpretation:** Errors are higher during peak periods, which is expected because queue behavior is more volatile. This identifies where predictions are less stable.

**Is this standard?** Yes. Segment-level error checks are a standard best practice for operational forecasting.

## 7. Feature Importance (Model Insight)

**Purpose:** Explain which inputs drive predictions most. This is helpful for interpretability and debugging.

**Top features (RandomForest):**

- queue_length_at_arrival: 0.2685
- waiting_time_lag1: 0.2656
- service_time_min: 0.1726
- queue_length_lag1: 0.1114
- is_peak_day: 0.0819

**Low-impact features:**

- is_holiday: 0.0006
- is_end_of_month: 0.0005
- is_pre_holiday: 0.0003

**Interpretation:** Real-time operational signals (queue length, lagged wait) dominate predictions, while calendar flags are minor contributors.

**Is this standard?** Yes. Feature importance reporting is standard for tree-based models.

## Summary: Are the Evaluations Standard?

Yes. This project uses a standard, production-appropriate evaluation stack for a tabular ML problem with time effects:

- Data quality checks
- Baseline comparison
- Multiple evaluation splits (random, chronological, CV)
- Tail error analysis (P90/P95)
- Segment error checks
- Feature importance reporting

The results show strong accuracy (MAE near 3 minutes, R2 around 0.96) and consistent performance across time-aware splits.
