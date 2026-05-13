import os
import sys
from pathlib import Path
from datetime import datetime
import re
from urllib.request import urlretrieve
import zipfile

import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from sklearn.ensemble import GradientBoostingRegressor

# Setup paths - Backend folder is root for deployment
BASE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASE_DIR.parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "models" / "queue_model.pkl"))
MODEL_URL = os.environ.get("MODEL_URL", None)


def _resolve_data_csv():
    env = os.environ.get("DATA_PATH")
    if env:
        return Path(env)
    candidates = [
        BASE_DIR / "data" / "synthetic_lto_cdo_queue_90days.csv",
        BASE_DIR / "Data" / "synthetic_lto_cdo_queue_90days.csv",
        PROJECT_ROOT / "data" / "synthetic_lto_cdo_queue_90days.csv",
        PROJECT_ROOT / "Data" / "synthetic_lto_cdo_queue_90days.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    return candidates[0]


def _resolve_holiday_calendar():
    name = "2026-calendar-with-holidays-portrait-sunday-start-en-ph.csv"
    env = os.environ.get("HOLIDAY_CALENDAR_PATH")
    if env:
        return Path(env)
    for p in (BASE_DIR / "data" / name, PROJECT_ROOT / "data" / name):
        if p.exists():
            return p
    for p in (BASE_DIR / "Data" / name, PROJECT_ROOT / "Data" / name):
        if p.exists():
            return p
    return BASE_DIR / "data" / name


def _outputs_dir():
    env = os.environ.get("OUTPUTS_DIR")
    if env:
        return Path(env)
    p = BASE_DIR / "outputs"
    if p.exists():
        return p
    return PROJECT_ROOT / "outputs"


DATA_PATH = _resolve_data_csv()
HOLIDAY_CALENDAR_PATH = _resolve_holiday_calendar()

# Add preprocessing module to path (stays in root src)
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import preprocessing functions
from Preprocessing.features import FEATURES, build_feature_dataframe

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model and data on startup
model = None
df = None
_hourly_chart_cache = None

_MODEL_KEY_DISPLAY = {
    "RandomForest": "Random Forest",
    "GradientBoosting": "Gradient Boosting",
    "LinearRegression": "Linear Regression",
}
_MODEL_CARD_NAME = {
    "RandomForest": "Random Forest Regressor",
    "GradientBoosting": "Gradient Boosting",
    "LinearRegression": "Linear Regression",
}
_MODEL_COLORS = {
    "RandomForest": "from-red-600 to-red-900",
    "GradientBoosting": "from-orange-600 to-orange-900",
    "LinearRegression": "from-blue-600 to-blue-900",
}
_FEATURE_CHART_COLORS = [
    "#EF4444",
    "#F97316",
    "#FBBF24",
    "#10B981",
    "#3B82F6",
    "#8B5CF6",
    "#EC4899",
    "#6366F1",
    "#14B8A6",
    "#A855F7",
    "#F43F5E",
    "#84CC16",
    "#0EA5E9",
    "#D946EF",
    "#78716C",
    "#64748B",
]


def _congestion_from_wait(wait_minutes):
    if wait_minutes > 45:
        return "HIGH", "AVOID - Very long queues (45+ min)"
    if wait_minutes > 25:
        return "MODERATE", "CAUTION - Moderate wait (25-45 min)"
    return "LOW", "GOOD - Short wait (<25 min)"


def load_model_and_data():
    """Load model and training data on startup"""
    global model, df, _hourly_chart_cache
    _hourly_chart_cache = None
    try:
        # Ensure models directory exists
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Download model from URL if it doesn't exist locally but URL is provided
        if not MODEL_PATH.exists() and MODEL_URL:
            print(f"â¬‡ï¸ Downloading model from {MODEL_URL}...")
            try:
                # Download to temp file
                download_path = MODEL_PATH.parent / "model_download.tmp"
                urlretrieve(MODEL_URL, download_path)
                
                # If it's a zip, extract it
                if str(download_path).endswith('.zip'):
                    print(f"ðŸ“¦ Extracting model...")
                    with zipfile.ZipFile(download_path, 'r') as zip_ref:
                        zip_ref.extractall(MODEL_PATH.parent)
                    download_path.unlink()  # Delete temp zip
                else:
                    # Rename temp file to model path
                    download_path.rename(MODEL_PATH)
                
                print(f"âœ… Model ready at {MODEL_PATH}")
            except Exception as e:
                print(f"âŒ Failed to download model: {e}")
        
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            print(f"âœ… Model loaded from {MODEL_PATH}")
        else:
            print(f"âš ï¸ Model not found at {MODEL_PATH}. Set MODEL_URL env var to a zip file URL or train the model first.")
            
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            df['date'] = pd.to_datetime(df['date'])
            if 'week_of_month' not in df.columns:
                df['week_of_month'] = ((df['date'].dt.day - 1) // 7 + 1).astype(int)
            print(f"âœ… Data loaded from {DATA_PATH}")
        else:
            print(f"âš ï¸ Data not found at {DATA_PATH}")
            
    except Exception as e:
        print(f"âŒ Error loading model/data: {e}")


def load_ph_holidays():
    """Load Philippine holidays from calendar"""
    month_map = {
        "Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
        "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12,
    }
    holiday_md = set()
    
    if not HOLIDAY_CALENDAR_PATH.exists():
        return holiday_md

    try:
        text = HOLIDAY_CALENDAR_PATH.read_text(encoding="utf-8", errors="ignore")
        for line in text.splitlines():
            match = re.search(r"\b([A-Za-z]{3})\s+(\d{1,2})\s*:\s*", line)
            if not match:
                continue
            month_name = match.group(1).title()
            day = int(match.group(2))
            month = month_map.get(month_name)
            if month:
                holiday_md.add((month, day))
    except Exception as e:
        print(f"âš ï¸ Error loading holidays: {e}")
    
    return holiday_md


def _read_model_comparison_csv():
    path = _outputs_dir() / "model_comparison.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _human_feature_name(raw):
    mapping = {
        "queue_length_at_arrival": "Queue at arrival",
        "waiting_time_lag1": "Wait time (lag)",
        "service_time_min": "Service time",
        "queue_length_lag1": "Queue length (lag)",
        "is_peak_day": "Peak day",
        "is_peak_hour": "Peak hour",
        "day_of_week": "Day of week",
        "hour": "Hour",
        "month_sin": "Month (sin)",
        "month_cos": "Month (cos)",
        "month": "Month",
        "is_weekend": "Weekend",
        "week_of_month": "Week of month",
        "is_holiday": "Holiday",
        "is_end_of_month": "End of month",
        "is_pre_holiday": "Pre-holiday",
    }
    return mapping.get(raw, raw.replace("_", " ").title())


def _build_model_performance_tables(cmp_df):
    """comparisonData + performanceMetrics from training outputs/model_comparison.csv."""
    if cmp_df is None or len(cmp_df) == 0:
        return [], []
    df_sorted = cmp_df.sort_values("robust_mae", ascending=True).reset_index(drop=True)
    best_model = str(df_sorted.iloc[0]["model"])
    comparison_data = []
    performance_metrics = []
    for _, r in df_sorted.iterrows():
        key = str(r["model"])
        if key == best_model:
            category = "Best"
        elif key == "LinearRegression":
            category = "Baseline"
        else:
            category = "Alternative"
        status = {
            "Best": "Best performer",
            "Baseline": "Baseline reference",
            "Alternative": "High reliability",
        }[category]
        comparison_data.append(
            {
                "model": _MODEL_KEY_DISPLAY.get(key, key),
                "mae": round(float(r["test_mae"]), 2),
                "rmse": round(float(r["test_rmse"]), 2),
                "r2": round(float(r["test_r2"]), 4),
                "category": category,
            }
        )
        performance_metrics.append(
            {
                "model": _MODEL_CARD_NAME.get(key, key),
                "metricType": "Waiting Time",
                "mae": round(float(r["test_mae"]), 2),
                "maeUnit": "mins",
                "accuracy": f"{int(round(float(r['test_r2']) * 100))}%",
                "status": status,
                "color": _MODEL_COLORS.get(key, "from-gray-600 to-gray-900"),
            }
        )
    return comparison_data, performance_metrics


def _hour_label(hour):
    labels = {8: "8am", 9: "9am", 10: "10am", 11: "11am", 12: "12pm", 13: "1pm", 14: "2pm", 15: "3pm", 16: "4pm"}
    return labels.get(int(hour), f"{int(hour)}")


def _build_hourly_prediction_chart():
    """
    Actual vs model predictions by hour of day.
    Random Forest uses the deployed joblib model; Gradient Boosting is fit once (cached)
    with the same hyperparameters as training for a comparable second curve.
    """
    global model, df, _hourly_chart_cache
    if _hourly_chart_cache is not None:
        return _hourly_chart_cache
    if df is None or not all(c in df.columns for c in FEATURES + ["waiting_time_min", "hour"]):
        return []

    plot_df = df[["waiting_time_min"] + FEATURES].copy()
    plot_df["actual"] = plot_df["waiting_time_min"]
    X = plot_df[FEATURES]

    if model is not None:
        plot_df["pred_rf"] = model.predict(X)
    else:
        plot_df["pred_rf"] = plot_df["actual"]

    if len(plot_df) > 0:
        try:
            gb_trees = int(os.environ.get("CHART_GB_N_ESTIMATORS", "100"))
            gb = GradientBoostingRegressor(
                n_estimators=max(20, gb_trees),
                learning_rate=0.05,
                max_depth=3,
                subsample=0.9,
                random_state=42,
            )
            gb.fit(X, plot_df["actual"])
            plot_df["pred_gb"] = gb.predict(X)
        except Exception as e:
            print(f"âš ï¸ GB chart model fit skipped: {e}")
            plot_df["pred_gb"] = plot_df["pred_rf"]
    else:
        plot_df["pred_gb"] = plot_df["pred_rf"]

    g = (
        plot_df.groupby("hour", as_index=False)
        .agg(actual=("actual", "mean"), predicted=("pred_rf", "mean"), gb=("pred_gb", "mean"))
        .sort_values("hour")
    )
    chart_data = []
    for _, row in g.iterrows():
        h = int(row["hour"])
        if h < 8 or h > 16:
            continue
        chart_data.append(
            {
                "hour": _hour_label(h),
                "actual": round(float(row["actual"]), 2),
                "predicted": round(float(row["predicted"]), 2),
                "gb": round(float(row["gb"]), 2),
            }
        )
    _hourly_chart_cache = chart_data
    return chart_data


def _feature_rows_from_model():
    """Normalized feature importances from the loaded RandomForest (production model)."""
    global model
    if model is None or not hasattr(model, "feature_importances_"):
        return None
    vals = np.asarray(model.feature_importances_, dtype=float)
    total = float(vals.sum()) or 1.0
    vals = vals / total
    order = np.argsort(-vals)
    rows = []
    for rank, idx in enumerate(order):
        raw = FEATURES[int(idx)]
        rows.append(
            {
                "feature": _human_feature_name(raw),
                "importance": float(vals[idx]),
                "color": _FEATURE_CHART_COLORS[rank % len(_FEATURE_CHART_COLORS)],
            }
        )
    return rows


def _parse_metrics_txt_feature_block():
    path = _outputs_dir() / "metrics.txt"
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    lines = text.splitlines()
    try:
        start = next(i for i, ln in enumerate(lines) if ln.strip().upper().startswith("FEATURE IMPORTANCE"))
    except StopIteration:
        return []
    out = []
    for ln in lines[start + 1 :]:
        ln = ln.strip()
        if not ln:
            break
        if ":" not in ln:
            break
        name, val = ln.split(":", 1)
        try:
            v = float(val.strip())
        except ValueError:
            continue
        out.append({"feature": _human_feature_name(name.strip()), "importance": v, "color": "#EF4444"})
    total = sum(x["importance"] for x in out) or 1.0
    for i, x in enumerate(sorted(out, key=lambda z: -z["importance"])):
        x["importance"] = float(x["importance"]) / total
        x["color"] = _FEATURE_CHART_COLORS[i % len(_FEATURE_CHART_COLORS)]
    return sorted(out, key=lambda z: -z["importance"])


def _insights_from_dataset():
    global df
    if df is None:
        return []
    by_hour = df.groupby("hour")["waiting_time_min"].mean()
    peak_h = int(by_hour.idxmax()) if len(by_hour) else 10
    mon = df[df["day_of_week"] == 0]["waiting_time_min"].mean()
    wed = df[df["day_of_week"] == 2]["waiting_time_min"].mean()
    if pd.isna(mon):
        mon = 0.0
    if pd.isna(wed):
        wed = 0.0
    pct = int(round((mon / wed - 1) * 100)) if wed and wed > 0 else 0
    hol = df[df["is_holiday"] == 1]["waiting_time_min"].mean() if "is_holiday" in df.columns else None
    reg = df[df["is_holiday"] == 0]["waiting_time_min"].mean() if "is_holiday" in df.columns else None
    if hol is not None and pd.isna(hol):
        hol = None
    if reg is not None and pd.isna(reg):
        reg = None
    hol_pct = int(round((hol / reg - 1) * 100)) if hol is not None and reg and reg > 0 else 0
    peak_mae_txt = ""
    mpath = _outputs_dir() / "metrics.txt"
    if mpath.exists():
        for ln in mpath.read_text(encoding="utf-8", errors="ignore").splitlines():
            if ln.strip().startswith("Peak Hour MAE:"):
                peak_mae_txt = ln.split(":", 1)[-1].strip()
                break
    insights = [
        {
            "icon": "schedule",
            "title": "Peak hours drive congestion",
            "description": f"Hour {peak_h}:00 has the highest average wait ({round(float(by_hour.loc[peak_h]), 1)} mins).",
        },
        {
            "icon": "calendar_month",
            "title": "Monday vs mid-week",
            "description": f"Mondays average {round(float(mon), 1)} mins vs Wednesday {round(float(wed), 1)} mins ({pct:+d}% vs Wed).",
        },
    ]
    if hol is not None and reg is not None:
        insights.append(
            {
                "icon": "payments",
                "title": "Holiday load",
                "description": f"Holiday-flagged days average {round(float(hol), 1)} mins vs {round(float(reg), 1)} mins on regular days ({hol_pct:+d}%).",
            }
        )
    if peak_mae_txt:
        insights.append(
            {
                "icon": "monitoring",
                "title": "Model error by segment",
                "description": f"Peak-hour MAE from the last training report: {peak_mae_txt} minutes.",
            }
        )
    else:
        insights.append(
            {
                "icon": "monitoring",
                "title": "Segmented error",
                "description": "Training reports segment-level MAE for peak days and peak hours to validate fairness.",
            }
        )
    return insights[:4]


# Load once when the module is imported so gunicorn workers initialize the model.
load_model_and_data()


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "model_loaded": model is not None,
        "data_loaded": df is not None,
        "timestamp": datetime.now().isoformat()
    }
    return jsonify(status), 200


@app.route('/predict', methods=['POST'])
def predict():
    """
    Predict queue waiting time
    
    Expected JSON input:
    {
        "date": "2026-05-13",
        "hour": 10,
        "day_of_week": "Wednesday",
        "queue_length_at_arrival": 5
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ["date", "hour", "day_of_week", "queue_length_at_arrival"]
        if not all(field in data for field in required_fields):
            return jsonify({
                "error": f"Missing required fields. Expected: {required_fields}"
            }), 400
        
        # Build feature DataFrame from input JSON
        X = build_feature_dataframe(data, holiday_calendar_path=HOLIDAY_CALENDAR_PATH)
        prediction = float(model.predict(X)[0])
        congestion, recommendation = _congestion_from_wait(prediction)
        
        return jsonify({
            "success": True,
            "prediction": prediction,
            "congestion": congestion,
            "recommendation": recommendation,
            "input": data,
            "unit": "minutes",
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Prediction failed: {str(e)}"
        }), 400


@app.route('/batch-predict', methods=['POST'])
def batch_predict():
    """
    Predict for multiple queue scenarios
    
    Expected JSON input:
    {
        "predictions": [
            {"date": "2026-05-13", "hour": 10, "day_of_week": "Wednesday", "queue_length_at_arrival": 5},
            ...
        ]
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    try:
        data = request.get_json()
        predictions_input = data.get("predictions", [])
        if not predictions_input:
            return jsonify({"error": "No predictions provided"}), 400

        # Build feature DataFrame for all items at once
        X = build_feature_dataframe(predictions_input, holiday_calendar_path=HOLIDAY_CALENDAR_PATH)
        preds = model.predict(X)
        results = []
        for item, pred in zip(predictions_input, preds):
            results.append({
                "input": item,
                "prediction": float(pred),
                "status": "success"
            })
        
        return jsonify({
            "success": True,
            "results": results,
            "total": len(predictions_input),
            "successful": sum(1 for r in results if r["status"] == "success"),
            "timestamp": datetime.now().isoformat()
        }), 200
        
    except Exception as e:
        return jsonify({
            "error": f"Batch prediction failed: {str(e)}"
        }), 400


@app.route('/info', methods=['GET'])
def info():
    """Get information about the service"""
    return jsonify({
        "service": "iQueue Prediction Backend",
        "version": "1.0.0",
        "description": "ML-powered queue waiting time prediction service",
        "endpoints": {
            "GET /health": "Health check",
            "POST /predict": "Single prediction",
            "POST /batch-predict": "Batch predictions",
            "GET /info": "Service information"
        }
    }), 200


@app.route('/api/model-performance', methods=['GET'])
def api_model_performance():
    """Metrics from outputs/model_comparison.csv; hourly chart from real RF + GB fit on dataset."""
    global df
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500

    cmp_df = _read_model_comparison_csv()
    if cmp_df is not None and len(cmp_df) > 0:
        comparison_data, performance_metrics = _build_model_performance_tables(cmp_df)
    else:
        comparison_data, performance_metrics = [], []

    chart_data = _build_hourly_prediction_chart()

    return jsonify(
        {
            "comparisonData": comparison_data,
            "performanceMetrics": performance_metrics,
            "chartData": chart_data,
        }
    )


@app.route('/api/feature-importance', methods=['GET'])
def api_feature_importance():
    """Feature importances from the loaded model, with metrics.txt as fallback."""
    importance_data = _feature_rows_from_model() or _parse_metrics_txt_feature_block()
    if not importance_data:
        return jsonify({"importanceData": [], "topInsights": _insights_from_dataset()})

    top_insights = _insights_from_dataset()
    return jsonify({"importanceData": importance_data, "topInsights": top_insights})


@app.route('/api/historical-analytics', methods=['GET'])
def api_historical_analytics():
    """Return historical analytics used for frontend charts"""
    global df
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500

    def _safe_mean(series, digits=1):
        if series is None or len(series) == 0:
            return 0.0
        val = series.mean()
        if pd.isna(val):
            return 0.0
        return round(float(val), digits)

    # Daily average by day of week (Mon-Sat)
    daily_data = []
    day_names = {0: 'Mon', 1: 'Tue', 2: 'Wed', 3: 'Thu', 4: 'Fri', 5: 'Sat'}
    for day_num in range(6):
        day_data = df[df['day_of_week'] == day_num]
        if len(day_data) > 0:
            avg_wait = _safe_mean(day_data['waiting_time_min'], 1)
            busy = day_num in [0, 4]
            daily_data.append({
                "day": day_names[day_num],
                "avgWait": avg_wait,
                "trend": "High" if avg_wait > 40 else "Medium" if avg_wait > 25 else "Low",
                "busiest": busy,
            })

    hourly_data = []
    for hour in range(8, 17):
        hour_data = df[df['hour'] == hour]
        if len(hour_data) > 0:
            avg_wait = _safe_mean(hour_data['waiting_time_min'], 1)
            hourly_data.append({"hour": _hour_label(hour), "wait": avg_wait})

    heatmap_data = []
    for day_num in range(6):
        day_name = day_names[day_num]
        day_df = df[df['day_of_week'] == day_num]
        morning = _safe_mean(day_df[day_df['hour'].isin([8, 9, 10, 11])]['waiting_time_min'], 1)
        afternoon = _safe_mean(day_df[day_df['hour'].isin([12, 13, 14])]['waiting_time_min'], 1)
        evening = _safe_mean(day_df[day_df['hour'].isin([15, 16])]['waiting_time_min'], 1)
        heatmap_data.append({"day": day_name, "morning": morning, "afternoon": afternoon, "evening": evening})

    mid = float(df[df['day_of_week'].isin([2, 3])]['waiting_time_min'].mean())
    mon_mean = float(df[df['day_of_week'] == 0]['waiting_time_min'].mean())
    mon_pct = int(round((mon_mean / mid - 1) * 100)) if mid and mid > 0 and not pd.isna(mid) else 0
    eom = float(df[df['is_end_of_month'] == 1]['waiting_time_min'].mean()) if 'is_end_of_month' in df.columns else None
    rest = float(df[df['is_end_of_month'] == 0]['waiting_time_min'].mean()) if 'is_end_of_month' in df.columns else None
    eom_pct = int(round((eom / rest - 1) * 100)) if eom is not None and rest and rest > 0 and not pd.isna(eom) else 0

    insights = [
        {
            "title": "Mondays vs mid-week",
            "desc": f"Monday average vs Tueâ€“Thu blend ({mon_pct:+d}%)" if mon_pct else "Monday vs mid-week workload",
            "value": f"{round(mon_mean if not pd.isna(mon_mean) else 0, 0)} mins",
        },
        {
            "title": "Late morning peak",
            "desc": "Average wait 10â€“11am window",
            "value": f"{round(_safe_mean(df[df['hour'].isin([10, 11])]['waiting_time_min'], 0), 0)} mins",
        },
        {
            "title": "Friday congestion",
            "desc": "Friday average wait",
            "value": f"{round(_safe_mean(df[df['day_of_week'] == 4]['waiting_time_min'], 0), 0)} mins",
        },
        {
            "title": "End-of-month effect",
            "desc": "Last few days of month vs other days" if eom_pct else "Within-month seasonality",
            "value": f"{eom_pct:+d}% vs other days" if eom_pct else "See heatmap",
        },
    ]

    return jsonify({
        "dailyData": daily_data,
        "hourlyData": hourly_data,
        "heatmapData": heatmap_data,
        "insights": insights,
    })


@app.route('/api/predictive-analytics', methods=['GET'])
def api_predictive_analytics():
    """Slot summaries from the deployed model on historical rows in each time window."""
    global df, model
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500

    def slot_stats(hours, label_low, label_high):
        sub = df[df["hour"].isin(hours)]
        if len(sub) == 0:
            return None, None, None
        if model is not None and all(c in sub.columns for c in FEATURES):
            p = model.predict(sub[FEATURES])
            lo, hi = int(np.percentile(p, 25)), int(np.percentile(p, 75))
            mean_p = float(np.mean(p))
        else:
            p = sub["waiting_time_min"].to_numpy()
            lo, hi = int(np.percentile(p, 25)), int(np.percentile(p, 75))
            mean_p = float(np.mean(p))
        std = float(np.std(p))
        conf = int(max(60, min(99, round((1 - min(std / (mean_p + 1e-6), 0.5)) * 100))))
        return (lo, hi), mean_p, conf

    morning = slot_stats([8, 9, 10, 11], None, None)
    afternoon = slot_stats([12, 13, 14, 15], None, None)
    evening = slot_stats([15, 16], None, None)

    def band(slot, color_base, rec):
        if slot[0] is None:
            return {
                "waitTime": "â€”",
                "congestion": "Medium",
                "confidence": 80,
                "recommendation": rec,
                "color": color_base,
            }
        (lo, hi), mean_p, conf = slot
        if mean_p < 25:
            cg = "Low"
        elif mean_p > 40:
            cg = "High"
        else:
            cg = "Medium"
        return {
            "waitTime": f"{lo}-{hi}",
            "congestion": cg,
            "confidence": conf,
            "recommendation": rec,
            "color": color_base,
        }

    predictions = {
        "morning": band(morning, "from-green-600", "8 AM - 11 AM typically shows the lowest modeled waits in this dataset."),
        "afternoon": band(afternoon, "from-orange-600", "12 PM - 3 PM is the busiest block; consider early morning if you need shorter waits."),
        "evening": band(evening, "from-yellow-600", "4 PM - 5 PM balances remaining service hours with queue length."),
    }

    time_slots = [
        {"id": "morning", "label": "Morning", "time": "8 AM - 12 PM"},
        {"id": "afternoon", "label": "Afternoon", "time": "12 PM - 4 PM"},
        {"id": "evening", "label": "Evening", "time": "4 PM - 6 PM"},
    ]

    waits = df["waiting_time_min"]
    operational = int(round((waits < 40).mean() * 100))
    slow = int(round(((waits >= 40) & (waits < 60)).mean() * 100))
    down = max(0, 100 - operational - slow)
    if down + slow + operational != 100:
        operational = 100 - slow - down

    return jsonify({
        "predictions": predictions,
        "timeSlots": time_slots,
        "systemReliability": {"operational": operational, "slow": slow, "down": down},
    })


@app.route('/api/dataset-summary', methods=['GET'])
def api_dataset_summary():
    global df
    if df is None:
        return jsonify({"error": "Data not loaded"}), 500

    return jsonify({
        "totalRecords": len(df),
        "dateRange": {"start": df['date'].min().strftime('%Y-%m-%d'), "end": df['date'].max().strftime('%Y-%m-%d')},
        "averageWaitTime": round(df['waiting_time_min'].mean(), 2),
        "medianWaitTime": round(df['waiting_time_min'].median(), 2),
        "maxWaitTime": round(df['waiting_time_min'].max(), 2),
        "peakHour": int(df.groupby('hour')['waiting_time_min'].mean().idxmax()),
        "busiestDay": int(df[df['day_of_week'] < 6].groupby('day_of_week')['waiting_time_min'].mean().idxmax()),
    })


@app.route('/api/metrics', methods=['GET'])
def api_metrics():
    """Return evaluation metrics from outputs/metrics.txt (raw + parsed feature importance)."""
    metrics_path = _outputs_dir() / "metrics.txt"
    if not metrics_path.exists():
        return jsonify({"error": "metrics file not found"}), 404

    text = metrics_path.read_text(encoding="utf-8")

    # Parse FEATURE IMPORTANCE section into structured list
    feature_importance = []
    try:
        lines = text.splitlines()
        start = next(i for i, l in enumerate(lines) if l.strip().upper().startswith("FEATURE IMPORTANCE"))
        for ln in lines[start + 1 :]:
            ln = ln.strip()
            if not ln:
                break
            if ":" in ln:
                name, val = ln.split(":", 1)
                try:
                    v = float(val.strip())
                except Exception:
                    v = val.strip()
                feature_importance.append({"feature": name.strip(), "importance": v})
            else:
                # stop parsing on unexpected format
                break
    except StopIteration:
        feature_importance = []

    return jsonify({"metricsText": text, "featureImportance": feature_importance}), 200


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    # Get port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    print(f"\nðŸš€ Starting iQueue Backend Service on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=debug)

