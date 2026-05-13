import os
import sys
from pathlib import Path
from datetime import datetime
import re
from urllib.request import urlretrieve

import joblib
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# Setup paths - Backend folder is root for deployment
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = Path(os.environ.get("MODEL_PATH", BASE_DIR / "models" / "queue_model.pkl"))
MODEL_URL = os.environ.get("MODEL_URL", None)
DATA_PATH = BASE_DIR / "data" / "synthetic_lto_cdo_queue_90days.csv"
HOLIDAY_CALENDAR_PATH = BASE_DIR / "data" / "2026-calendar-with-holidays-portrait-sunday-start-en-ph.csv"

# Add preprocessing module to path (stays in root src)
PROJECT_ROOT = BASE_DIR.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Import preprocessing functions
from preprocess import get_features

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Load model and data on startup
model = None
df = None


def load_model_and_data():
    """Load model and training data on startup"""
    global model, df
    try:
        # Ensure models directory exists
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        
        # Download model from URL if it doesn't exist locally but URL is provided
        if not MODEL_PATH.exists() and MODEL_URL:
            print(f"⬇️ Downloading model from {MODEL_URL}...")
            try:
                urlretrieve(MODEL_URL, MODEL_PATH)
                print(f"✅ Model downloaded to {MODEL_PATH}")
            except Exception as e:
                print(f"❌ Failed to download model: {e}")
        
        if MODEL_PATH.exists():
            model = joblib.load(MODEL_PATH)
            print(f"✅ Model loaded from {MODEL_PATH}")
        else:
            print(f"⚠️ Model not found at {MODEL_PATH}. Set MODEL_URL env var or train the model first.")
            
        if DATA_PATH.exists():
            df = pd.read_csv(DATA_PATH)
            df['date'] = pd.to_datetime(df['date'])
            print(f"✅ Data loaded from {DATA_PATH}")
        else:
            print(f"⚠️ Data not found at {DATA_PATH}")
            
    except Exception as e:
        print(f"❌ Error loading model/data: {e}")


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
        print(f"⚠️ Error loading holidays: {e}")
    
    return holiday_md


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
        
        # Parse date
        date_obj = pd.to_datetime(data["date"])
        
        # Create feature dictionary
        features = {
            "date": date_obj,
            "hour": int(data["hour"]),
            "day_of_week": data["day_of_week"],
            "queue_length_at_arrival": float(data["queue_length_at_arrival"])
        }
        
        # Get features (same preprocessing as training)
        X = get_features([features])
        
        # Make prediction
        prediction = model.predict(X)[0]
        
        return jsonify({
            "success": True,
            "prediction": float(prediction),
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
        
        results = []
        for item in predictions_input:
            try:
                date_obj = pd.to_datetime(item["date"])
                features = {
                    "date": date_obj,
                    "hour": int(item["hour"]),
                    "day_of_week": item["day_of_week"],
                    "queue_length_at_arrival": float(item["queue_length_at_arrival"])
                }
                
                X = get_features([features])
                pred = model.predict(X)[0]
                
                results.append({
                    "input": item,
                    "prediction": float(pred),
                    "status": "success"
                })
            except Exception as e:
                results.append({
                    "input": item,
                    "error": str(e),
                    "status": "failed"
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


@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    return jsonify({"error": "Internal server error"}), 500


if __name__ == '__main__':
    load_model_and_data()
    
    # Get port from environment or default to 5000
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') != 'production'
    
    print(f"\n🚀 Starting iQueue Backend Service on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=debug)
