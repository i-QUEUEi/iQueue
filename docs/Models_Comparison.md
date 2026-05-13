# Model Comparison: Why Random Forest

This project benchmarks `LinearRegression`, `GradientBoosting`, and `RandomForest`. RandomForest is selected because it delivers the best accuracy and stability across all evaluation splits. Results come from [outputs/metrics.txt](../outputs/metrics.txt) and [outputs/model_comparison.csv](../outputs/model_comparison.csv).

---

## Where Models Are Defined

| Model | Builder file | Key parameters |
|---|---|---|
| `LinearRegression` | [linear_regression.py — L4–5](../src/model_implementation/model_zoo/linear_regression.py#L4-L5) | Default sklearn settings |
| `RandomForest` | [random_forest.py — L5–12](../src/model_implementation/model_zoo/random_forest.py#L5-L12) | 500 trees, max_depth=15, max_features="sqrt" |
| `GradientBoosting` | [gradient_boosting.py — L5–11](../src/model_implementation/model_zoo/gradient_boosting.py#L5-L11) | 250 trees, learning_rate=0.05, subsample=0.9 |

All three are assembled into one dictionary at [model_zoo/\_\_init\_\_.py — L6–11](../src/model_implementation/model_zoo/__init__.py#L6-L11) and looped over in [train_model.py — L57–90](../src/model_implementation/train_model.py#L57-L90).

---

## Head-to-Head Results

Each model is evaluated using **3 methods** inside [`evaluate_model()`](../src/Evaluation/evaluation.py#L33):
- Random split — [evaluation.py — L45–49](../src/Evaluation/evaluation.py#L45-L49)
- Chronological split — [evaluation.py — L51–53](../src/Evaluation/evaluation.py#L51-L53) using [splits.py — L4–23](../src/Evaluation/splits.py#L4-L23)
- 5-fold cross-validation — [evaluation.py — L55–66](../src/Evaluation/evaluation.py#L55-L66)

Then averaged into `robust_mae` at [evaluation.py — L75](../src/Evaluation/evaluation.py#L75).

| Model | Test MAE | Chrono MAE | CV MAE | **Robust MAE** | Test R² |
|---|---|---|---|---|---|
| **RandomForest ✅** | 2.92 | 2.74 | 2.89 | **2.85** | 0.9637 |
| GradientBoosting | 3.09 | 2.93 | 3.00 | 3.01 | 0.9613 |
| LinearRegression | 4.44 | 4.12 | 4.39 | 4.32 | 0.9248 |

**What the metrics mean:**
- **MAE** — average error in minutes. Lower = better.
- **R²** — how much of the wait-time variation the model explains. Closer to 1 = better.
- **Robust MAE** — average of all three MAEs. Reflects stability, not just one lucky split.

The winner is picked at [train_model.py — L89–90](../src/model_implementation/train_model.py#L89-L90):
```python
if selected_result is None or result["robust_mae"] < selected_result["robust_mae"]:
    selected_result = result
```

---

## Why RandomForest Works Best Here

Written to `metrics.txt` at [reporting.py — L60–63](../src/Evaluation/reporting.py#L60-L63).

1. **Nonlinear patterns** — Queue dynamics are not linear. RandomForest captures interactions between time, queue length, and lag features naturally.
2. **Handles mixed feature types** — Works with numeric + binary flags without manual scaling.
3. **Stable under noise** — 500 trees ([random_forest.py — L6](../src/model_implementation/model_zoo/random_forest.py#L6)) reduce variance and handle synthetic noise better than a single model.
4. **Consistent generalization** — Best on both random and chronological splits, proving it works on unseen future dates.

---

## Why the Other Models Lose

Written to `metrics.txt` at [reporting.py — L65–68](../src/Evaluation/reporting.py#L65-L68).

- **`LinearRegression`** — Too simple. It draws one straight line through all the data. Queue wait times don't follow a straight line (Monday 10am is much worse than Wednesday 10am, even at the same queue length). This causes a MAE of 4.44 vs RandomForest's 2.92.
- **`GradientBoosting`** — Competitive but slightly weaker. It uses only 250 trees ([gradient_boosting.py — L6](../src/model_implementation/model_zoo/gradient_boosting.py#L6)) with a slow learning rate, which limits capacity on this dataset. Its robust MAE of 3.01 is still 6% worse than RandomForest.

---

## Tradeoffs

| Concern | Reality |
|---|---|
| Training speed | RandomForest is slower than LinearRegression, but still finishes in seconds |
| Interpretability | Lower than linear, but feature importance is available at [evaluation.py — L10–30](../src/Evaluation/evaluation.py#L10-L30) |
| Overfitting risk | Controlled by `max_depth=15`, `min_samples_leaf=2` at [random_forest.py — L7–9](../src/model_implementation/model_zoo/random_forest.py#L7-L9) |

---

## Final Decision

RandomForest is saved as the model artifact at [train_model.py — L110](../src/model_implementation/train_model.py#L110):
```python
joblib.dump(selected_result["model"], MODEL_PATH)
```
It is then loaded at prediction time by [context.py — L21](../src/Prediction/context.py#L21):
```python
model = joblib.load(MODEL_PATH)
```

It wins because it has the lowest robust MAE (2.85), the highest R² (0.9637), and the most consistent performance across all three evaluation methods.
