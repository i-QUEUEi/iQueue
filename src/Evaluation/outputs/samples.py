import numpy as np
import pandas as pd


def sample_predictions(model, features):
    """Run 6 hardcoded sanity-check predictions to verify the model learned correctly.

    Each test case has a KNOWN expected answer from the TRUE_PATTERNS table.
    If the model's predictions are wildly off, something went wrong in training.

    The test cases cover extremes:
    - Monday 9-10am (busiest slots in the week) — should predict HIGH
    - Wednesday 8am (calmest slot in the week) — should predict LOW

    Args:
        model: The trained winning model.
        features: List of 16 feature column names (for DataFrame column order).
    """
    print("\n🎯 SAMPLE PREDICTIONS (Using actual patterns):")

    # 6 test cases with known expected answers from TRUE_PATTERNS
    test_cases = [
        {"name": "Monday 9am (Should be 55min)", "hour": 9, "day_of_week": 0, "is_peak_day": 1, "is_peak_hour": 1, "queue": 25, "lag_wait": 25},
        {"name": "Monday 10am (Should be 70min)", "hour": 10, "day_of_week": 0, "is_peak_day": 1, "is_peak_hour": 1, "queue": 35, "lag_wait": 35},
        {"name": "Monday 8am (Should be 25min)", "hour": 8, "day_of_week": 0, "is_peak_day": 1, "is_peak_hour": 0, "queue": 8, "lag_wait": 15},
        {"name": "Wednesday 9am (Should be 19min)", "hour": 9, "day_of_week": 2, "is_peak_day": 0, "is_peak_hour": 1, "queue": 10, "lag_wait": 10},
        {"name": "Wednesday 10am (Should be 25min)", "hour": 10, "day_of_week": 2, "is_peak_day": 0, "is_peak_hour": 1, "queue": 12, "lag_wait": 12},
        {"name": "Wednesday 8am (Should be 9min)", "hour": 8, "day_of_week": 2, "is_peak_day": 0, "is_peak_hour": 0, "queue": 4, "lag_wait": 8},
    ]

    # Use January as a neutral test month (no seasonal extremes)
    sample_month = 1
    month_angle = 2 * np.pi * (sample_month - 1) / 12

    # Run each test case through the model
    for case in test_cases:
        # Manually construct all 16 features (same order as FEATURES list)
        feature_values = {
            "hour": case["hour"],
            "day_of_week": case["day_of_week"],
            "week_of_month": 2,                          # Mid-month (neutral, no end-of-month effect)
            "month": sample_month,
            "month_sin": float(np.sin(month_angle)),      # Cyclical encoding for January
            "month_cos": float(np.cos(month_angle)),
            "is_end_of_month": 0,                         # Not end of month
            "is_holiday": 0,                              # Not a holiday
            "is_pre_holiday": 0,                          # Not day before a holiday
            "is_peak_day": case["is_peak_day"],
            "queue_length_at_arrival": case["queue"],
            "service_time_min": 35,                       # Default service time
            "is_weekend": 1 if case["day_of_week"] == 5 else 0,
            "is_peak_hour": case["is_peak_hour"],
            "queue_length_lag1": max(2, case["queue"] - 3),  # Estimated previous queue
            "waiting_time_lag1": case["lag_wait"],            # Previous person's wait time
        }

        # Build a 1-row DataFrame in the exact feature order the model expects
        row = [feature_values[name] for name in features]
        test_X = pd.DataFrame([row], columns=features)

        # Get prediction and print comparison with expected value
        pred = model.predict(test_X)[0]
        print(f"   {case['name']}: {pred:.1f} min")
