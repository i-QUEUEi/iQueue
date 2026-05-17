# iQueue — Presentation Requirements Audit

> [!IMPORTANT]
> All 5 requirements are addressed. Req #3 meets the minimum (3 models). See note at bottom.

---

## 1. ✅ Robust Evaluation

**File:** [evaluation.py](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/Evaluation/model_quality/model_evaluation.py)

Three separate evaluation strategies are combined into a single **`robust_mae`** score:

| Strategy | Lines | What it does |
|---|---|---|
| Random split | [L48–49](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/Evaluation/model_quality/model_evaluation.py#L48-L49) | Trains on `X_train`, tests on `X_test` |
| Chronological split | [L51–53](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/Evaluation/model_quality/model_evaluation.py#L51-L53) | Trains on earliest 80% of dates, tests on latest 20% |
| 5-Fold Cross-Validation | [L55–66](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/Evaluation/model_quality/model_evaluation.py#L55-L66) | Shuffled KFold, scoring MAE + RMSE + R² |

The three MAEs are then averaged into one score:
```python
# evaluation.py — Line 75
robust_mae = float(np.mean([test_metrics["mae"], chrono_metrics["mae"], cv_mae]))
```

Error percentiles are also captured at [L77–80](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/Evaluation/model_quality/model_evaluation.py#L77-L80):
```python
p90_abs_error = float(np.percentile(abs_errors, 90))
p95_abs_error = float(np.percentile(abs_errors, 95))
max_abs_error = float(np.max(abs_errors))
```

The chronological split logic lives in [splits.py](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/Evaluation/model_quality/splits.py) — sorts by date, then cuts at 80% of unique dates ([L9–16](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/Evaluation/model_quality/splits.py#L9-L16)).

---

## 2. ✅ Data Evaluation

**File:** [model_evaluation.py — L137–155](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/Evaluation/data_quality/data_evaluation.py-L155)

`evaluate_data_quality()` inspects the raw CSV before any preprocessing:

```python
# evaluation.py — Lines 141–155
return {
    "rows": int(len(raw_df)),
    "columns": int(raw_df.shape[1]),
    "duplicate_rows": int(raw_df.duplicated().sum()),
    "missing_cells": int(raw_df.isna().sum().sum()),
    "negative_waiting_rows": int((target < 0).sum()),
    "negative_queue_rows": int((queue < 0).sum()),
    "target_mean": float(target.mean()),
    "target_median": float(target.median()),
    "target_std": float(target.std()),
    "target_min": float(target.min()),
    "target_p10": float(target.quantile(0.10)),
    "target_p90": float(target.quantile(0.90)),
    "target_max": float(target.max()),
}
```

This is called in [train_model.py — L45](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/model_implementation/train_model.py#L45) on the **raw** dataframe (before cleaning), and the results are written to `outputs/metrics.txt` via [reporting.py — L11–22](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/Evaluation/outputs/reporting.py#L11-L22).

---

## 3. ⚠️ 3–4 Models (Currently: 3)

**File:** [model_zoo/\_\_init\_\_.py](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/model_implementation/model_zoo/__init__.py)

```python
# model_zoo/__init__.py — Lines 6–11
def build_model_catalog(random_state):
    return {
        "LinearRegression": build_linear_regression(),
        "RandomForest": build_random_forest(random_state),
        "GradientBoosting": build_gradient_boosting(random_state),
    }
```

Individual model files:
- [linear_regression.py](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/model_implementation/model_zoo/linear_regression.py)
- [random_forest.py](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/model_implementation/model_zoo/random_forest.py)
- [gradient_boosting.py](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/model_implementation/model_zoo/gradient_boosting.py)

> [!WARNING]
> You have **3 models** which meets the minimum. The requirement says "3–4 models." Adding a 4th (e.g. Ridge Regression) would fully cover the range.

---

## 4. ✅ Data Visualization

**File:** [plots.py](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/Evaluation/outputs/plots.py)

| Plot | Function | Lines | Output file |
|---|---|---|---|
| Waiting time distribution | `plot_target_distribution()` | [L8–16](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/Evaluation/outputs/plots.py#L8-L16) | `outputs/plots/target_distribution.png` |
| Day × Hour heatmap | `plot_day_hour_heatmap()` | [L19–37](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/Evaluation/outputs/plots.py#L19-L37) | `outputs/plots/day_hour_heatmap.png` |
| Model comparison bar chart | `plot_model_comparison()` | [L40–56](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/Evaluation/outputs/plots.py#L40-L56) | `outputs/plots/model_comparison.png` |
| Actual vs Predicted scatter | `plot_actual_vs_predicted()` | [L59–69](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/Evaluation/outputs/plots.py#L59-L69) | `outputs/plots/actual_vs_predicted.png` |

All 4 are triggered in [train_model.py — L113–116](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/model_implementation/train_model.py#L113-L116).

---

## 5. ✅ Why and Why Not (Models + Preprocessing Justification)

**File:** [reporting.py — L60–73](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/Evaluation/outputs/reporting.py#L60-L73)

Three dedicated sections are written to `outputs/metrics.txt`:

```python
# reporting.py — Lines 60–73
f.write("\nWHY THESE MODELS\n")
f.write("LinearRegression is the simplest baseline...")
f.write("RandomForest handles nonlinear interactions...")
f.write("GradientBoosting can capture residual structure...")

f.write("\nWHY NOT OTHER OPTIONS\n")
f.write("We did not rely on only a mean predictor...")
f.write("We did not add heavy preprocessing like scaling...")
f.write("We avoided overly complex models such as neural networks...")

f.write("\nPREPROCESSING CHOICES\n")
f.write("Parsed dates and derived week_of_month...")
f.write("Filtered negative waiting times and queue lengths...")
f.write("Kept the engineered lag and peak features...")
```

Feature importance rankings are also printed right after at [L75–77](file:///c:/Users/Rhenel%20Jhon%20Sajol/Documents/IQUEUE/iQueue/src/Evaluation/outputs/reporting.py#L75-L77), further backing up the "why these features" story.

---

## Summary

| # | Requirement | Status |
|---|---|---|
| 1 | Robust evaluation | ✅ Random + Chronological + CV → `robust_mae` |
| 2 | Data evaluation | ✅ `evaluate_data_quality()` on raw data |
| 3 | 3–4 models | ⚠️ 3/4 — add one more to be safe |
| 4 | Data visualization | ✅ 4 plots saved to `outputs/plots/` |
| 5 | Why and why not | ✅ Written to `metrics.txt` with 3 sections |



