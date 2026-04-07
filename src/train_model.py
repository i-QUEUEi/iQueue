import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from preprocess import load_data, get_features

# Resolve paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_lto_cdo_queue_90days.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "queue_model.pkl")

RF_PARAMS = {
    "n_estimators": 500,
    "max_depth": 15,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "random_state": 42,
}


def compute_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }

# LOAD DATA
df = load_data(DATA_PATH)

X, y, features = get_features(df)

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Use Random Forest with more trees and depth to capture patterns
model = RandomForestRegressor(**RF_PARAMS)

model.fit(X_train, y_train)

# PREDICTIONS
y_train_pred = model.predict(X_train)
y_pred = model.predict(X_test)

# CORE METRICS
train_metrics = compute_metrics(y_train, y_train_pred)
test_metrics = compute_metrics(y_test, y_pred)

print("\n📊 MODEL PERFORMANCE")
print(f"Train MAE: {train_metrics['mae']:.2f} minutes")
print(f"Train RMSE: {train_metrics['rmse']:.2f} minutes")
print(f"Train R2 Score: {train_metrics['r2']:.4f}")
print(f"Test MAE: {test_metrics['mae']:.2f} minutes")
print(f"Test RMSE: {test_metrics['rmse']:.2f} minutes")
print(f"Test R2 Score: {test_metrics['r2']:.4f}")

# BASELINE COMPARISON (predict mean training wait time for every case)
baseline_value = y_train.mean()
baseline_train_pred = np.full(y_train.shape, baseline_value, dtype=float)
baseline_test_pred = np.full(y_test.shape, baseline_value, dtype=float)

baseline_train_metrics = compute_metrics(y_train, baseline_train_pred)
baseline_test_metrics = compute_metrics(y_test, baseline_test_pred)

mae_improvement_pct = (
    (baseline_test_metrics["mae"] - test_metrics["mae"]) / baseline_test_metrics["mae"]
) * 100
rmse_improvement_pct = (
    (baseline_test_metrics["rmse"] - test_metrics["rmse"]) / baseline_test_metrics["rmse"]
) * 100

# OVERFITTING CHECK
mae_gap = test_metrics["mae"] - train_metrics["mae"]
r2_gap = train_metrics["r2"] - test_metrics["r2"]

# WORST-CASE ERROR BEHAVIOR
abs_errors = np.abs(y_test.to_numpy() - y_pred)
p90_abs_error = np.percentile(abs_errors, 90)
p95_abs_error = np.percentile(abs_errors, 95)
max_abs_error = np.max(abs_errors)

# SEGMENT-WISE ERRORS
day_name_map = {
    0: "Monday",
    1: "Tuesday",
    2: "Wednesday",
    3: "Thursday",
    4: "Friday",
    5: "Saturday",
}
day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

eval_df = X_test.copy().reset_index(drop=True)
eval_df["actual_wait"] = y_test.reset_index(drop=True)
eval_df["pred_wait"] = y_pred
eval_df["abs_error"] = np.abs(eval_df["actual_wait"] - eval_df["pred_wait"])
eval_df["day_name"] = eval_df["day_of_week"].map(day_name_map)

day_error = eval_df.groupby("day_name")["abs_error"].agg(["mean", "median", "max", "count"])
day_error = day_error.reindex(day_order).dropna()

hour_error = (
    eval_df.groupby("hour")["abs_error"]
    .agg(["mean", "median", "max", "count"])
    .sort_index()
)

peak_day_error = eval_df.groupby("is_peak_day")["abs_error"].mean()
peak_hour_error = eval_df.groupby("is_peak_hour")["abs_error"].mean()

# TIME-AWARE VALIDATION (first 80% dates train, last 20% dates test)
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

time_model = RandomForestRegressor(**RF_PARAMS)
time_model.fit(X_time_train, y_time_train)
y_time_pred = time_model.predict(X_time_test)
time_metrics = compute_metrics(y_time_test, y_time_pred)

# UNCERTAINTY COVERAGE CHECK using tree-level prediction spread (P10-P90)
# Individual trees are fitted internally on arrays, so use array input here to avoid feature-name warnings.
X_test_array = X_test.to_numpy()
tree_predictions = np.vstack([tree.predict(X_test_array) for tree in model.estimators_])
test_p10 = np.percentile(tree_predictions, 10, axis=0)
test_p90 = np.percentile(tree_predictions, 90, axis=0)

within_band = (y_test.to_numpy() >= test_p10) & (y_test.to_numpy() <= test_p90)
coverage_p10_p90 = np.mean(within_band)
avg_band_width = np.mean(test_p90 - test_p10)

# Feature importance
print("\n🔍 FEATURE IMPORTANCE:")
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.iterrows():
    print(f"   {row['feature']}: {row['importance']:.4f}")

# Test predictions for key scenarios (using actual patterns)
print("\n🎯 SAMPLE PREDICTIONS (Using actual patterns):")
test_cases = [
    {'name': 'Monday 9am (Should be 55min)', 'hour': 9, 'day_of_week': 0, 'is_peak_day': 1, 'is_peak_hour': 1, 'queue': 25, 'lag_wait': 25},
    {'name': 'Monday 10am (Should be 70min)', 'hour': 10, 'day_of_week': 0, 'is_peak_day': 1, 'is_peak_hour': 1, 'queue': 35, 'lag_wait': 35},
    {'name': 'Monday 8am (Should be 25min)', 'hour': 8, 'day_of_week': 0, 'is_peak_day': 1, 'is_peak_hour': 0, 'queue': 8, 'lag_wait': 15},
    {'name': 'Wednesday 9am (Should be 19min)', 'hour': 9, 'day_of_week': 2, 'is_peak_day': 0, 'is_peak_hour': 1, 'queue': 10, 'lag_wait': 10},
    {'name': 'Wednesday 10am (Should be 25min)', 'hour': 10, 'day_of_week': 2, 'is_peak_day': 0, 'is_peak_hour': 1, 'queue': 12, 'lag_wait': 12},
    {'name': 'Wednesday 8am (Should be 9min)', 'hour': 8, 'day_of_week': 2, 'is_peak_day': 0, 'is_peak_hour': 0, 'queue': 4, 'lag_wait': 8},
]

for case in test_cases:
    test_X = pd.DataFrame([[
        case['hour'],
        case['day_of_week'],
        2,   # week_of_month (use week 2 as representative mid-month)
        case['is_peak_day'],
        case['queue'],
        35,  # avg service time
        0,   # not weekend
        case['is_peak_hour'],
        max(2, case['queue'] - 3),  # lag queue
        case['lag_wait']  # lag wait
    ]], columns=features)
    
    pred = model.predict(test_X)[0]
    print(f"   {case['name']}: {pred:.1f} min")

# SAVE MODEL
joblib.dump(model, MODEL_PATH)
print(f"\n✅ Model saved to {MODEL_PATH}")

# SAVE METRICS
OUTPUTS_PATH = os.path.join(BASE_DIR, "outputs", "metrics.txt")
os.makedirs(os.path.dirname(OUTPUTS_PATH), exist_ok=True)
with open(OUTPUTS_PATH, "w") as f:
    f.write("CORE METRICS (RANDOM TEST SPLIT)\n")
    f.write(f"Train MAE: {train_metrics['mae']:.2f}\n")
    f.write(f"Train RMSE: {train_metrics['rmse']:.2f}\n")
    f.write(f"Train R2 Score: {train_metrics['r2']:.4f}\n")
    f.write(f"Test MAE: {test_metrics['mae']:.2f}\n")
    f.write(f"Test RMSE: {test_metrics['rmse']:.2f}\n")
    f.write(f"Test R2 Score: {test_metrics['r2']:.4f}\n")

    f.write("\nBASELINE COMPARISON (MEAN PREDICTOR)\n")
    f.write(f"Baseline Train MAE: {baseline_train_metrics['mae']:.2f}\n")
    f.write(f"Baseline Train RMSE: {baseline_train_metrics['rmse']:.2f}\n")
    f.write(f"Baseline Train R2 Score: {baseline_train_metrics['r2']:.4f}\n")
    f.write(f"Baseline Test MAE: {baseline_test_metrics['mae']:.2f}\n")
    f.write(f"Baseline Test RMSE: {baseline_test_metrics['rmse']:.2f}\n")
    f.write(f"Baseline Test R2 Score: {baseline_test_metrics['r2']:.4f}\n")
    f.write(f"Test MAE Improvement vs Baseline: {mae_improvement_pct:.2f}%\n")
    f.write(f"Test RMSE Improvement vs Baseline: {rmse_improvement_pct:.2f}%\n")

    f.write("\nOVERFITTING CHECK\n")
    f.write(f"MAE Gap (Test - Train): {mae_gap:.2f}\n")
    f.write(f"R2 Gap (Train - Test): {r2_gap:.4f}\n")

    f.write("\nWORST-CASE ABSOLUTE ERROR\n")
    f.write(f"P90 Absolute Error: {p90_abs_error:.2f}\n")
    f.write(f"P95 Absolute Error: {p95_abs_error:.2f}\n")
    f.write(f"Max Absolute Error: {max_abs_error:.2f}\n")

    f.write("\nTIME-AWARE VALIDATION (CHRONOLOGICAL SPLIT)\n")
    f.write(f"Chronological Train Dates: {len(time_train_dates)}\n")
    f.write(f"Chronological Test Dates: {len(time_test_dates)}\n")
    f.write(f"Chronological Test MAE: {time_metrics['mae']:.2f}\n")
    f.write(f"Chronological Test RMSE: {time_metrics['rmse']:.2f}\n")
    f.write(f"Chronological Test R2 Score: {time_metrics['r2']:.4f}\n")

    f.write("\nUNCERTAINTY COVERAGE (TREE P10-P90 BAND ON TEST SET)\n")
    f.write(f"P10-P90 Coverage: {coverage_p10_p90 * 100:.2f}%\n")
    f.write(f"Average P10-P90 Width: {avg_band_width:.2f}\n")

    f.write("\nSEGMENT ERRORS - BY PEAK FLAGS (TEST SET MAE)\n")
    f.write(f"Peak Day MAE (Mon/Fri): {peak_day_error.get(1, np.nan):.2f}\n")
    f.write(f"Non-Peak Day MAE: {peak_day_error.get(0, np.nan):.2f}\n")
    f.write(f"Peak Hour MAE: {peak_hour_error.get(1, np.nan):.2f}\n")
    f.write(f"Non-Peak Hour MAE: {peak_hour_error.get(0, np.nan):.2f}\n")

    f.write("\nSEGMENT ERRORS - BY DAY (TEST SET)\n")
    for day, row in day_error.iterrows():
        f.write(
            f"{day}: MAE={row['mean']:.2f}, MedianAE={row['median']:.2f}, MaxAE={row['max']:.2f}, N={int(row['count'])}\n"
        )

    f.write("\nSEGMENT ERRORS - BY HOUR (TEST SET)\n")
    for hour, row in hour_error.iterrows():
        f.write(
            f"{int(hour):02d}:00 - MAE={row['mean']:.2f}, MedianAE={row['median']:.2f}, MaxAE={row['max']:.2f}, N={int(row['count'])}\n"
        )

    f.write("\nFeature Importance:\n")
    for idx, row in feature_importance.iterrows():
        f.write(f"{row['feature']}: {row['importance']:.4f}\n")