# Deploying iQueue Backend to Render

## Quick Setup (5 minutes)

### 1. Prepare Your Repository
```bash
# Ensure these files are in Backend/ directory:
# - app.py (Flask API)
# - requirements.txt (Python dependencies)
# - Procfile (for Render)
```

### 2. Create Render Service

1. Go to [render.com](https://render.com)
2. Sign in with GitHub
3. Click **New +** → **Web Service**
4. Connect your GitHub repo (iQueue)
5. Fill in these settings:

| Setting | Value |
|---------|-------|
| **Name** | `iqueue-api` (or your choice) |
| **Environment** | Python |
| **Region** | Choose closest to you |
| **Branch** | `main` |
| **Build Command** | `pip install -r Backend/requirements.txt` |
| **Start Command** | `gunicorn Backend.app:app` |
| **Plan** | Free (starts with free tier) |

### 3. Environment Variables
In Render dashboard, add to your service:
```
FLASK_ENV = production
PORT = 5000
```

### 4. Deploy
Click **Deploy** and wait ~3-5 minutes. You'll get a URL like:
```
https://iqueue-api.onrender.com
```

---

## Testing Your API

### Health Check
```bash
curl https://iqueue-api.onrender.com/health
```

### Single Prediction
```bash
curl -X POST https://iqueue-api.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "date": "2026-05-13",
    "hour": 10,
    "day_of_week": "Wednesday",
    "queue_length_at_arrival": 5
  }'
```

### Batch Predictions
```bash
curl -X POST https://iqueue-api.onrender.com/batch-predict \
  -H "Content-Type: application/json" \
  -d '{
    "predictions": [
      {"date": "2026-05-13", "hour": 10, "day_of_week": "Wednesday", "queue_length_at_arrival": 5},
      {"date": "2026-05-14", "hour": 14, "day_of_week": "Thursday", "queue_length_at_arrival": 8}
    ]
  }'
```

---

## Available Endpoints

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/health` | GET | Check if API is running |
| `/info` | GET | API documentation |
| `/predict` | POST | Single prediction |
| `/batch-predict` | POST | Multiple predictions |

---

## Frontend Integration Example

### JavaScript/React
```javascript
const API_URL = 'https://iqueue-api.onrender.com';

async function predictQueueTime(date, hour, dayOfWeek, queueLength) {
  const response = await fetch(`${API_URL}/predict`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      date,
      hour,
      day_of_week: dayOfWeek,
      queue_length_at_arrival: queueLength
    })
  });
  
  return await response.json();
}

// Usage
const result = await predictQueueTime('2026-05-13', 10, 'Wednesday', 5);
console.log(`Estimated wait: ${result.prediction} minutes`);
```

---

## Troubleshooting

**Build fails:**
- Check `Backend/requirements.txt` has all dependencies
- Make sure model file exists at `models/queue_model.pkl`

**API returns 500 error:**
- Check Render logs: Dashboard → Service → Logs
- Ensure model is trained before deploying

**CORS issues with frontend:**
- Already configured with `Flask-CORS`
- Verify frontend is calling correct API URL

---

## Local Testing Before Deploy

```bash
# Install dependencies
pip install -r Backend/requirements.txt

# Run locally
cd Backend
python app.py
```

Then visit: `http://localhost:5000/health`

---

## Tips

✅ **Free tier limits:** 15-minute auto-sleep with inactivity
✅ **To keep it running:** Use a monitoring service or upgrade to paid plan
✅ **Model updates:** Retrain model, push to GitHub, Render auto-deploys
✅ **Logs:** View on Render dashboard for debugging
