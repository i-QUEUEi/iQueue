import numpy as np


def chronological_split(df, features):
    df_time = df.sort_values("date").copy()
    normalized_dates = df_time["date"].dt.normalize()
    unique_dates = np.sort(normalized_dates.unique())

    split_idx = int(len(unique_dates) * 0.8)
    split_idx = min(max(split_idx, 1), len(unique_dates) - 1)

    time_train_dates = unique_dates[:split_idx]
    time_test_dates = unique_dates[split_idx:]

    time_train_df = df_time[normalized_dates.isin(time_train_dates)]
    time_test_df = df_time[normalized_dates.isin(time_test_dates)]

    X_time_train = time_train_df[features]
    y_time_train = time_train_df["waiting_time_min"]
    X_time_test = time_test_df[features]
    y_time_test = time_test_df["waiting_time_min"]

    return (X_time_train, y_time_train), (X_time_test, y_time_test), len(time_train_dates), len(time_test_dates)
