# Model Comparison: Why Random Forest

This project benchmarks `LinearRegression`, `GradientBoosting`, and `RandomForest`. RandomForest is selected because it delivers the best accuracy and stability across all evaluation splits. The results below come from [outputs/metrics.txt](outputs/metrics.txt) and [outputs/model_comparison.csv](outputs/model_comparison.csv).

## Head-to-Head Results (Key Metrics)

- RandomForest: Test MAE 2.92, Chrono MAE 2.74, CV MAE 2.89, Robust MAE 2.85, Test R2 0.9637
- GradientBoosting: Test MAE 3.09, Chrono MAE 2.93, CV MAE 3.00, Robust MAE 3.01, Test R2 0.9613
- LinearRegression: Test MAE 4.44, Chrono MAE 4.12, CV MAE 4.39, Robust MAE 4.32, Test R2 0.9248

**Why these numbers matter:**

- `MAE` is the average error in minutes. Lower is better.
- `R2` measures how much variance the model explains. Closer to 1 is better.
- `Robust MAE` averages multiple evaluation splits, so it reflects stability, not just one lucky split.

RandomForest has the lowest robust MAE and the strongest R2, so it is both accurate and consistent.

## Why RandomForest Works Best Here

1. **Nonlinear patterns:** Queue dynamics are not strictly linear. RandomForest captures nonlinear interactions between time, queue length, and lag features.
2. **Handles mixed feature types:** It works well with a mix of numeric and binary flags without manual scaling.
3. **Stable under noise:** Tree ensembles reduce variance and handle the synthetic noise in the dataset better than a single model.
4. **Strong generalization:** It performs best on both random splits and chronological splits, which shows good time-aware generalization.

## Why the Other Models Lose

- `LinearRegression` is too simple for the nonlinear patterns, leading to much higher error.
- `GradientBoosting` is competitive but slightly less stable and slightly higher error in all splits.

## Tradeoffs (Why It Is Still Acceptable)

- RandomForest is heavier than LinearRegression, but training time is still practical.
- Interpretability is lower than linear models, but feature importance is available and sufficient for this project.

## Final Decision

RandomForest is chosen because it delivers the best accuracy, strongest stability across time-aware evaluation, and robust performance on noisy queue data. It is the most reliable option for a beginner-friendly, real-world queue prediction tool.
