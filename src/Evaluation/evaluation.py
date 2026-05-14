import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import KFold, cross_validate

from .metrics import compute_metrics


def get_feature_importance(model, features, X_reference, y_reference, random_state):
    """Determine how much each feature contributes to the model's predictions.

    Uses three strategies depending on the model type:
    1. Tree-based models (RF, GB): use built-in feature_importances_
    2. Linear models: use absolute coefficient values
    3. Other models: use permutation importance (shuffle each feature and measure damage)

    Args:
        model: A trained sklearn model.
        features: List of 16 feature names.
        X_reference, y_reference: Test data to evaluate importance against.
        random_state: Seed for reproducibility.

    Returns:
        DataFrame with columns ['feature', 'importance'], sorted highest → lowest,
        with importances normalized to sum to 1.0.
    """
    if hasattr(model, "feature_importances_"):
        # Random Forest / Gradient Boosting: each tree tracks which features
        # it used most to reduce prediction error
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        # Linear Regression: coefficients tell how much each feature shifts
        # the prediction. We take absolute values since sign doesn't matter here.
        values = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    else:
        # Fallback: randomly shuffle one feature at a time and measure
        # how much worse the model gets. Big drop = important feature.
        importance = permutation_importance(
            model,
            X_reference,
            y_reference,
            n_repeats=10,
            random_state=random_state,
            scoring="neg_mean_absolute_error",
        )
        values = importance.importances_mean

    # Build a DataFrame and normalize so all importances sum to 1.0
    importance_df = pd.DataFrame({"feature": features, "importance": values})
    total = importance_df["importance"].sum()
    if total > 0:
        importance_df["importance"] = importance_df["importance"] / total
    return importance_df.sort_values("importance", ascending=False)


def evaluate_model(
    name,
    model,
    X_train,
    X_test,
    y_train,
    y_test,
    chrono_train,
    chrono_test,
    features,
    random_state,
):
    """Run a complete 3-way evaluation of a single model.

    This is the core evaluation function. For each model it:
    1. Trains on random split → tests on random split (Way 1)
    2. Trains on oldest 80% → tests on newest 20% (Way 2: chronological)
    3. Runs 5-fold cross-validation (Way 3)
    4. Combines all three into a robust_mae score
    5. Computes segment errors, percentile errors, and feature importance

    Args:
        name: Model name string (e.g., "RandomForest").
        model: An untrained sklearn model object.
        X_train, X_test, y_train, y_test: Random 80/20 split data.
        chrono_train, chrono_test: Chronological split data (tuples of X, y).
        features: List of 16 feature column names.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary containing the trained model, all predictions, all metrics,
        segment error breakdowns, and feature importance rankings.
    """
    # ===== WAY 1: Random Split =====
    # clone() creates a fresh untrained copy — ensures no leftover state
    fitted_model = clone(model)
    fitted_model.fit(X_train, y_train)         # Train on 80% of randomly split data

    train_pred = fitted_model.predict(X_train)  # Predict on training data (to check overfitting)
    test_pred = fitted_model.predict(X_test)    # Predict on unseen 20% (the real test)

    # ===== WAY 2: Chronological Split =====
    # A completely separate model trained only on old data, tested on new data
    chrono_model = clone(model)
    chrono_model.fit(chrono_train[0], chrono_train[1])    # Train on oldest 80% of dates
    chrono_pred = chrono_model.predict(chrono_test[0])    # Predict on newest 20%

    # ===== WAY 3: 5-Fold Cross-Validation =====
    # Split data into 5 chunks. Train on 4, test on 1. Repeat 5 times.
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        clone(model),       # Fresh model for each fold
        X_train,
        y_train,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",       # Sklearn uses negative (higher = better)
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
    )

    # ===== Compute metrics for all 3 ways =====
    train_metrics = compute_metrics(y_train, train_pred)     # Training accuracy (overfitting check)
    test_metrics = compute_metrics(y_test, test_pred)        # Random split test accuracy
    chrono_metrics = compute_metrics(chrono_test[1], chrono_pred)  # Chronological test accuracy

    # Flip sklearn's negative scores back to positive
    cv_mae = float(-cv_results["test_mae"].mean())
    cv_rmse = float(-cv_results["test_rmse"].mean())
    cv_r2 = float(cv_results["test_r2"].mean())

    # ===== Robust MAE: Average of all 3 evaluation methods =====
    # A model must score well on ALL 3 to get a low robust_mae
    robust_mae = float(np.mean([test_metrics["mae"], chrono_metrics["mae"], cv_mae]))

    # ===== Percentile error analysis =====
    # Shows the distribution of errors, not just the average
    abs_errors = np.abs(y_test.to_numpy() - test_pred)
    p90_abs_error = float(np.percentile(abs_errors, 90))    # 90% of predictions within this
    p95_abs_error = float(np.percentile(abs_errors, 95))    # 95% of predictions within this
    max_abs_error = float(np.max(abs_errors))                # Worst single prediction ever

    # ===== Segment error analysis =====
    # Check if the model is biased against certain days/hours
    eval_df = X_test.copy().reset_index(drop=True)
    eval_df["actual_wait"] = y_test.reset_index(drop=True)
    eval_df["pred_wait"] = test_pred
    eval_df["abs_error"] = np.abs(eval_df["actual_wait"] - eval_df["pred_wait"])

    # Map day numbers back to names for readable output
    day_name_map = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
    }
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    eval_df["day_name"] = eval_df["day_of_week"].map(day_name_map)

    # Error breakdown by day of week (is the model worse on Mondays?)
    day_error = (
        eval_df.groupby("day_name")["abs_error"]
        .agg(["mean", "median", "max", "count"])
        .reindex(day_order)
        .dropna()
    )
    # Error breakdown by hour (is the model worse at 10am?)
    hour_error = eval_df.groupby("hour")["abs_error"].agg(["mean", "median", "max", "count"]).sort_index()
    # Error on peak days (Mon/Fri) vs non-peak days
    peak_day_error = eval_df.groupby("is_peak_day")["abs_error"].mean()
    # Error during peak hours vs non-peak hours
    peak_hour_error = eval_df.groupby("is_peak_hour")["abs_error"].mean()

    # ===== Feature importance =====
    # Which features matter most to this model?
    feature_importance = get_feature_importance(fitted_model, features, X_test, y_test, random_state)

    # ===== Return everything in a single dictionary =====
    return {
        "name": name,
        "model": fitted_model,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "chrono_pred": chrono_pred,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "chrono_metrics": chrono_metrics,
        "cv_mae": cv_mae,
        "cv_rmse": cv_rmse,
        "cv_r2": cv_r2,
        "robust_mae": robust_mae,
        "baseline_value": float(np.mean(y_train)),
        "p90_abs_error": p90_abs_error,
        "p95_abs_error": p95_abs_error,
        "max_abs_error": max_abs_error,
        "day_error": day_error,
        "hour_error": hour_error,
        "peak_day_error": peak_day_error,
        "peak_hour_error": peak_hour_error,
        "feature_importance": feature_importance,
        "eval_df": eval_df,
        "chrono_train_size": int(len(chrono_train[0])),
        "chrono_test_size": int(len(chrono_test[0])),
    }


def evaluate_data_quality(raw_df):
    """Audit the raw CSV for data quality issues BEFORE training begins.

    Computes 14 statistics about the dataset to catch problems early:
    row count, duplicates, missing values, impossible values, and
    distributional stats for the target variable.

    Args:
        raw_df: The raw DataFrame read directly from CSV (before any cleaning).

    Returns:
        Dictionary of 14 quality metrics.
    """
    target = raw_df["waiting_time_min"]
    queue = raw_df["queue_length_at_arrival"]

    return {
        "rows": int(len(raw_df)),                          # Total number of records
        "columns": int(raw_df.shape[1]),                   # Total number of columns
        "duplicate_rows": int(raw_df.duplicated().sum()),  # Exact duplicate records
        "missing_cells": int(raw_df.isna().sum().sum()),   # Total blank cells
        "negative_waiting_rows": int((target < 0).sum()),  # Physically impossible waits
        "negative_queue_rows": int((queue < 0).sum()),     # Physically impossible queues
        "target_mean": float(target.mean()),               # Average wait time
        "target_median": float(target.median()),           # Middle wait time (50th percentile)
        "target_std": float(target.std()),                 # Spread/variation of wait times
        "target_min": float(target.min()),                 # Shortest wait recorded
        "target_p10": float(target.quantile(0.10)),        # 10th percentile (short waits)
        "target_p90": float(target.quantile(0.90)),        # 90th percentile (long waits)
        "target_max": float(target.max()),                 # Longest wait recorded
    }
