import pandas as pd
import joblib
import os
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from preprocess import load_data, get_features

# Resolve paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "synthetic_lto_cdo_queue_90days.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "queue_model.pkl")

# LOAD DATA
df = load_data(DATA_PATH)

X, y, features = get_features(df)

# SPLIT DATA
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Use Random Forest with more trees and depth to capture patterns
model = RandomForestRegressor(
    n_estimators=500,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    random_state=42
)

model.fit(X_train, y_train)

# PREDICTIONS
y_pred = model.predict(X_test)

# METRICS
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n📊 MODEL PERFORMANCE")
print(f"MAE: {mae:.2f} minutes")
print(f"RMSE: {rmse:.2f} minutes")
print(f"R2 Score: {r2:.4f}")

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
    f.write(f"MAE: {mae:.2f}\n")
    f.write(f"RMSE: {rmse:.2f}\n")
    f.write(f"R2 Score: {r2:.4f}\n")
    f.write("\nFeature Importance:\n")
    for idx, row in feature_importance.iterrows():
        f.write(f"{row['feature']}: {row['importance']:.4f}\n")