"""model_evaluation.py — Robust 3-way model evaluation.

Evaluates a trained model three different ways and combines the results
into a single robust_mae score used to select the winner:
  1. Random 80/20 split
  2. Chronological split (oldest 80% → newest 20%)
  3. 5-Fold Cross-Validation
"""
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
    3. Other models: use permutation importance

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
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    else:
        importance = permutation_importance(
            model, X_reference, y_reference,
            n_repeats=10, random_state=random_state,
            scoring="neg_mean_absolute_error",
        )
        values = importance.importances_mean

    importance_df = pd.DataFrame({"feature": features, "importance": values})
    total = importance_df["importance"].sum()
    if total > 0:
        importance_df["importance"] = importance_df["importance"] / total
    return importance_df.sort_values("importance", ascending=False)


def evaluate_model(name, model, X_train, X_test, y_train, y_test,
                   chrono_train, chrono_test, features, random_state):
    """Run a complete 3-way robust evaluation of a single model.

    Args:
        name: Model name string (e.g., "RandomForest").
        model: An untrained sklearn model object.
        X_train, X_test, y_train, y_test: Random 80/20 split data.
        chrono_train, chrono_test: Chronological split data (tuples of X, y).
        features: List of 16 feature column names.
        random_state: Seed for reproducibility.

    Returns:
        Dictionary with the trained model, all predictions, all metrics,
        segment error breakdowns, and feature importance rankings.
    """
    # WAY 1 — Random Split
    fitted_model = clone(model)
    fitted_model.fit(X_train, y_train)
    train_pred = fitted_model.predict(X_train)
    test_pred = fitted_model.predict(X_test)

    # WAY 2 — Chronological Split
    chrono_model = clone(model)
    chrono_model.fit(chrono_train[0], chrono_train[1])
    chrono_pred = chrono_model.predict(chrono_test[0])

    # WAY 3 — 5-Fold Cross-Validation
    cv = KFold(n_splits=5, shuffle=True, random_state=random_state)
    cv_results = cross_validate(
        clone(model), X_train, y_train, cv=cv,
        scoring={"mae": "neg_mean_absolute_error",
                 "rmse": "neg_root_mean_squared_error",
                 "r2": "r2"},
    )

    train_metrics = compute_metrics(y_train, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)
    chrono_metrics = compute_metrics(chrono_test[1], chrono_pred)

    cv_mae = float(-cv_results["test_mae"].mean())
    cv_rmse = float(-cv_results["test_rmse"].mean())
    cv_r2 = float(cv_results["test_r2"].mean())

    # Robust MAE: average of all 3 methods — must score well on ALL three
    robust_mae = float(np.mean([test_metrics["mae"], chrono_metrics["mae"], cv_mae]))

    # Percentile errors
    abs_errors = np.abs(y_test.to_numpy() - test_pred)
    p90_abs_error = float(np.percentile(abs_errors, 90))
    p95_abs_error = float(np.percentile(abs_errors, 95))
    max_abs_error = float(np.max(abs_errors))

    # Segment errors
    eval_df = X_test.copy().reset_index(drop=True)
    eval_df["actual_wait"] = y_test.reset_index(drop=True)
    eval_df["pred_wait"] = test_pred
    eval_df["abs_error"] = np.abs(eval_df["actual_wait"] - eval_df["pred_wait"])

    day_name_map = {0: "Monday", 1: "Tuesday", 2: "Wednesday",
                    3: "Thursday", 4: "Friday", 5: "Saturday"}
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    eval_df["day_name"] = eval_df["day_of_week"].map(day_name_map)

    day_error = (eval_df.groupby("day_name")["abs_error"]
                 .agg(["mean", "median", "max", "count"])
                 .reindex(day_order).dropna())
    hour_error = (eval_df.groupby("hour")["abs_error"]
                  .agg(["mean", "median", "max", "count"]).sort_index())
    peak_day_error = eval_df.groupby("is_peak_day")["abs_error"].mean()
    peak_hour_error = eval_df.groupby("is_peak_hour")["abs_error"].mean()

    feature_importance = get_feature_importance(fitted_model, features, X_test, y_test, random_state)

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
