"""data_evaluation.py — Data quality audit (Step 1 of training pipeline).

Checks the raw CSV for problems BEFORE any preprocessing or training begins.
Catches issues like duplicates, missing values, and impossible wait times.
"""


def evaluate_data_quality(raw_df):
    """Audit the raw CSV for data quality issues BEFORE training begins.

    Computes 13 statistics about the dataset to catch problems early:
    row count, duplicates, missing values, impossible values, and
    distributional stats for the target variable.

    Args:
        raw_df: The raw DataFrame read directly from CSV (before any cleaning).

    Returns:
        Dictionary of 13 quality metrics.
    """
    target = raw_df["waiting_time_min"]
    queue = raw_df["queue_length_at_arrival"]

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
