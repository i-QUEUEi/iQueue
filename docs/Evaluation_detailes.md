# Evaluation Details

This document explains every evaluation step used in this project, what each metric means, the results from the latest run, and whether the evaluation approach is standard. Metrics and values below come from [outputs/metrics.txt](../outputs/metrics.txt) and [outputs/model_comparison.csv](../outputs/model_comparison.csv).

---

## 1. Data Quality Evaluation (Input Sanity Checks)

**Purpose:** Confirm the dataset is clean and suitable for training. This is a standard prerequisite for any ML project.

**Where in the code:**
- Function: [`evaluate_data_quality()`](../src/Evaluation/evaluation.py#L137) in [evaluation.py](../src/Evaluation/evaluation.py)
- Called on the **raw** CSV at [train_model.py — L45](../src/model_implementation/train_model.py#L45)
- Results written to `metrics.txt` at [reporting.py — L11–22](../src/Evaluation/reporting.py#L11-L22)

**What each check does:**

| Check | Code line | Result |
|---|---|---|
| Row count | [L142](../src/Evaluation/evaluation.py#L142) | 21,428 rows |
| Column count | [L143](../src/Evaluation/evaluation.py#L143) | 20 columns |
| Duplicate rows | [L144](../src/Evaluation/evaluation.py#L144) | 0 |
| Missing cells | [L145](../src/Evaluation/evaluation.py#L145) | 0 |
| Negative waiting rows | [L146](../src/Evaluation/evaluation.py#L146) | 0 |
| Negative queue rows | [L147](../src/Evaluation/evaluation.py#L147) | 0 |

**Target distribution (waiting_time_min)** — computed at [L148–155](../src/Evaluation/evaluation.py#L148-L155):

| Stat | Value |
|---|---|
| Mean | 36.56 min |
| Median | 28.60 min |
| Std deviation | 22.05 min |
| Min / Max | 6.60 / 90.00 min |
| P10 / P90 | 14.90 / 73.60 min |

**Is this standard?** Yes. Basic integrity checks and target distribution summaries are standard and necessary in every supervised learning pipeline.

---

## 2. Model Benchmarking (Model-to-Model Comparison)

**Purpose:** Compare multiple model families and select the best one. Standard practice for tabular ML.

**Where in the code:**
- Model catalog: [model_zoo/\_\_init\_\_.py — L6–11](../src/model_implementation/model_zoo/__init__.py#L6-L11)
- Training loop: [train_model.py — L57–90](../src/model_implementation/train_model.py#L57-L90)
- Each model evaluated via [`evaluate_model()`](../src/Evaluation/evaluation.py#L33) in [evaluation.py — L33–134](../src/Evaluation/evaluation.py#L33-L134)
- Results sorted and printed at [train_model.py — L92–99](../src/model_implementation/train_model.py#L92-L99)

**Models compared:**

| Model | File | Key setting |
|---|---|---|
| `LinearRegression` | [linear_regression.py — L4–5](../src/model_implementation/model_zoo/linear_regression.py#L4-L5) | Default (no tuning) |
| `RandomForest` | [random_forest.py — L5–12](../src/model_implementation/model_zoo/random_forest.py#L5-L12) | 500 trees, max_depth=15 |
| `GradientBoosting` | [gradient_boosting.py — L5–11](../src/model_implementation/model_zoo/gradient_boosting.py#L5-L11) | 250 trees, lr=0.05 |

**Results:**

| Model | Test MAE | Chrono MAE | CV MAE | Robust MAE | Test R² |
|---|---|---|---|---|---|
| **RandomForest** ✅ | 2.92 | 2.74 | 2.89 | **2.85** | 0.9637 |
| GradientBoosting | 3.09 | 2.93 | 3.00 | 3.01 | 0.9613 |
| LinearRegression | 4.44 | 4.12 | 4.39 | 4.32 | 0.9248 |

**Why the selected model wins:** RandomForest has the lowest `robust_mae` — computed at [evaluation.py — L75](../src/Evaluation/evaluation.py#L75):
```python
robust_mae = float(np.mean([test_metrics["mae"], chrono_metrics["mae"], cv_mae]))
```
The winner is picked at [train_model.py — L89–90](../src/model_implementation/train_model.py#L89-L90).

**Is this standard?** Yes. Benchmarking multiple models by a stable metric is standard.

---

## 3. Baseline Comparison

**Purpose:** Ensure the model is better than a trivial predictor (always guessing the average).

**Where in the code:**
- Baseline built at [train_model.py — L101–103](../src/model_implementation/train_model.py#L101-L103)
- `compute_metrics()` used: [metrics.py — L5–10](../src/Evaluation/metrics.py#L5-L10)
- Written to report at [reporting.py — L36–39](../src/Evaluation/reporting.py#L36-L39)

**Baseline results (mean predictor):**

| Metric | Value |
|---|---|
| Test MAE | 18.18 min |
| Test RMSE | 22.28 min |
| Test R² | −0.0002 |

**Interpretation:** The baseline performs ~6× worse than RandomForest (MAE 18.18 vs 2.92). The model is genuinely learning patterns.

**Is this standard?** Yes. A mean predictor baseline is the standard sanity check.

---

## 4. Robust Evaluation (Multiple Splits)

**Purpose:** Make sure performance is not an accident of one split. All three methods run inside [`evaluate_model()`](../src/Evaluation/evaluation.py#L33).

### Random Split
- Code: [evaluation.py — L45–49](../src/Evaluation/evaluation.py#L45-L49)
- Split done in [train_model.py — L50](../src/model_implementation/train_model.py#L50): 80% train / 20% test, randomly shuffled

| Metric | RandomForest |
|---|---|
| MAE | 2.92 min |
| RMSE | 4.24 min |
| R² | 0.9637 |

### Chronological Split
- Split logic: [splits.py — L4–23](../src/Evaluation/splits.py#L4-L23) — sorts by date, cuts at 80% of unique dates
- Called at [train_model.py — L51](../src/model_implementation/train_model.py#L51)
- Trains `chrono_model` at [evaluation.py — L51–53](../src/Evaluation/evaluation.py#L51-L53)

| Metric | RandomForest |
|---|---|
| Train dates | 176 |
| Test dates | 44 |
| MAE | 2.74 min |
| RMSE | 4.03 min |
| R² | 0.9618 |

### 5-Fold Cross-Validation
- Code: [evaluation.py — L55–66](../src/Evaluation/evaluation.py#L55-L66)
- Uses `KFold(n_splits=5, shuffle=True)` scoring MAE, RMSE, and R²

| Metric | RandomForest |
|---|---|
| CV MAE | 2.89 min |
| CV RMSE | 4.19 min |
| CV R² | 0.9636 |

### Robust MAE (final score)
All three MAEs averaged at [evaluation.py — L75](../src/Evaluation/evaluation.py#L75):
```python
robust_mae = float(np.mean([test_metrics["mae"], chrono_metrics["mae"], cv_mae]))
# RandomForest: (2.92 + 2.74 + 2.89) / 3 = 2.85
```

**Is this standard?** Yes. Using all three together is a strong and responsible evaluation practice.

---

## 5. Error Distribution (Tail Risk)

**Purpose:** Understand how large errors get in worst cases, not just on average.

**Where in the code:** [evaluation.py — L77–80](../src/Evaluation/evaluation.py#L77-L80)
```python
abs_errors = np.abs(y_test.to_numpy() - test_pred)
p90_abs_error = float(np.percentile(abs_errors, 90))
p95_abs_error = float(np.percentile(abs_errors, 95))
max_abs_error = float(np.max(abs_errors))
```
Written to report at [reporting.py — L50–52](../src/Evaluation/reporting.py#L50-L52).

**Results (RandomForest):**

| Percentile | Error |
|---|---|
| P90 | 7.20 min |
| P95 | 9.78 min |
| Max | 25.23 min |

**Interpretation:** 90% of predictions are within 7 minutes. Only the worst rare cases reach ~25 minutes.

**Is this standard?** Yes. Percentile error reporting is standard for operational forecasting.

---

## 6. Segment Error Checks (Peak vs Non-Peak)

**Purpose:** Confirm performance does not collapse on busy periods.

**Where in the code:** [evaluation.py — L97–105](../src/Evaluation/evaluation.py#L97-L105)
```python
day_error   = eval_df.groupby("day_name")["abs_error"].agg(...)
hour_error  = eval_df.groupby("hour")["abs_error"].agg(...)
peak_day_error  = eval_df.groupby("is_peak_day")["abs_error"].mean()
peak_hour_error = eval_df.groupby("is_peak_hour")["abs_error"].mean()
```
Written to report at [reporting.py — L54–58](../src/Evaluation/reporting.py#L54-L58).

**Results (RandomForest):**

| Segment | MAE |
|---|---|
| Peak day (Mon/Fri) | 4.66 min |
| Non-peak day | 1.76 min |
| Peak hour (9–11am, 1–3pm) | 3.70 min |
| Non-peak hour | 1.95 min |

**Interpretation:** Errors are higher during peak periods because queue behavior is more volatile then. This is expected and acceptable.

**Is this standard?** Yes. Segment-level error checks are a standard best practice for operational forecasting.

---

## 7. Feature Importance (Model Insight)

**Purpose:** Explain which inputs drive predictions most.

**Where in the code:** [`get_feature_importance()`](../src/Evaluation/evaluation.py#L10) at [evaluation.py — L10–30](../src/Evaluation/evaluation.py#L10-L30). For RandomForest, it reads `model.feature_importances_` directly ([L11–12](../src/Evaluation/evaluation.py#L11-L12)). Results normalized to sum to 1 at [L27–29](../src/Evaluation/evaluation.py#L27-L29).

Written to report at [reporting.py — L75–77](../src/Evaluation/reporting.py#L75-L77).

**Top features (RandomForest):**

| Feature | Importance | Meaning |
|---|---|---|
| `queue_length_at_arrival` | 0.2685 | How many people are ahead of you right now |
| `waiting_time_lag1` | 0.2656 | How long the previous person waited |
| `service_time_min` | 0.1726 | Average time each transaction takes |
| `queue_length_lag1` | 0.1114 | Queue length one transaction ago |
| `is_peak_day` | 0.0819 | Whether it's Monday or Friday |

**Low-impact features:**

| Feature | Importance |
|---|---|
| `is_holiday` | 0.0006 |
| `is_end_of_month` | 0.0005 |
| `is_pre_holiday` | 0.0003 |

**Interpretation:** Real-time queue signals (current queue, lag features) dominate. Calendar flags matter very little.

**Is this standard?** Yes. Feature importance is standard for tree-based models.

---

## Summary

| Evaluation Type | Code location | Standard? |
|---|---|---|
| Data quality checks | [evaluation.py — L137–155](../src/Evaluation/evaluation.py#L137-L155) | ✅ Yes |
| Baseline comparison | [train_model.py — L101–103](../src/model_implementation/train_model.py#L101-L103) | ✅ Yes |
| Random split | [evaluation.py — L45–49](../src/Evaluation/evaluation.py#L45-L49) | ✅ Yes |
| Chronological split | [splits.py — L4–23](../src/Evaluation/splits.py#L4-L23) | ✅ Yes |
| Cross-validation | [evaluation.py — L55–66](../src/Evaluation/evaluation.py#L55-L66) | ✅ Yes |
| Tail error (P90/P95) | [evaluation.py — L77–80](../src/Evaluation/evaluation.py#L77-L80) | ✅ Yes |
| Segment error checks | [evaluation.py — L97–105](../src/Evaluation/evaluation.py#L97-L105) | ✅ Yes |
| Feature importance | [evaluation.py — L10–30](../src/Evaluation/evaluation.py#L10-L30) | ✅ Yes |

**Overall result:** MAE ≈ 3 minutes, R² ≈ 0.96, consistent across all three evaluation strategies.
