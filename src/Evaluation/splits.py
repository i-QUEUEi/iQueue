import numpy as np


def chronological_split(df, features):
    """Split data by date order: oldest 80% for training, newest 20% for testing.

    Unlike a random split, this ensures the model is tested on FUTURE dates
    it has never seen — simulating real-world deployment where you can only
    predict the future, not look it up.

    The split is by unique dates (not rows), so a busy Monday with 120 rows
    and a quiet Wednesday with 80 rows each count as one "date."

    Args:
        df: Full DataFrame with a 'date' column.
        features: List of feature column names (the 16 FEATURES).

    Returns:
        (X_train, y_train): Training data (oldest 80% of dates).
        (X_test, y_test): Test data (newest 20% of dates).
        len(train_dates): How many unique dates are in the training set.
        len(test_dates): How many unique dates are in the test set.
    """
    # Sort all rows by date (oldest first)
    df_time = df.sort_values("date").copy()

    # Get a list of unique dates (ignoring time-of-day)
    normalized_dates = df_time["date"].dt.normalize()
    unique_dates = np.sort(normalized_dates.unique())

    # Find the 80% cutoff point
    split_idx = int(len(unique_dates) * 0.8)
    # Safety: ensure at least 1 date in train and 1 in test
    split_idx = min(max(split_idx, 1), len(unique_dates) - 1)

    # Split dates into training (oldest) and testing (newest) groups
    time_train_dates = unique_dates[:split_idx]    # e.g., Jan–Aug
    time_test_dates = unique_dates[split_idx:]     # e.g., Sep–Nov

    # Select rows belonging to each group
    time_train_df = df_time[normalized_dates.isin(time_train_dates)]
    time_test_df = df_time[normalized_dates.isin(time_test_dates)]

    # Extract features (X) and target (y) for each split
    X_time_train = time_train_df[features]
    y_time_train = time_train_df["waiting_time_min"]
    X_time_test = time_test_df[features]
    y_time_test = time_test_df["waiting_time_min"]

    return (X_time_train, y_time_train), (X_time_test, y_time_test), len(time_train_dates), len(time_test_dates)
