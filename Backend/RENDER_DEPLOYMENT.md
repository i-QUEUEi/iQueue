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

### 4. Upload Model File
After deployment, upload your model:

1. Go to Render Dashboard → Select your service
2. Click **Shell** tab (at the top)
3. Run this command in the shell:
```bash
mkdir -p Backend/models
```

4. Then upload the model file:
   - Go to **Files** tab
   - Upload `queue_model.pkl` to `Backend/models/`
   - Or use SCP if you have SSH access

**Alternative (if shell upload doesn't work):**
```bash
# From your local terminal, after getting SSH access:
scp queue_model.pkl user@render-instance:/app/Backend/models/
```

### 5. Restart Service
After uploading, restart the service:
1. Go to Render Dashboard
2. Click the service
3. Click **Restart** button
4. Wait for it to come back online

### 4. Deploy
### 4. Deploy
Click **Deploy** and wait ~3-5 minutes. You'll get a URL like:
```
https://iqueue-api.onrender.com
```

### 5. Verify Model is Loaded
After uploading the model file and restarting:
```bash
curl https://iqueue-api.onrender.com/health
```

Should return:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "data_loaded": true,
  "timestamp": "..."
}
```

If `model_loaded` is `false`, check Render logs for errors.

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

**Model file not found (model_loaded: false):**
- Check model was uploaded to `Backend/models/queue_model.pkl`
- Verify file path in shell: `ls -la Backend/models/`
- Restart service after uploading

**Build fails:**
- Check `Backend/requirements.txt` has all dependencies
- View Render build logs for specific errors

**API returns 500 error:**
- Check Render logs: Dashboard → Service → Logs
- Ensure model is uploaded and service was restarted
- Verify data files exist if needed

**Model predicts but returns NaN/errors:**
- Check preprocessor can find `src/preprocess.py`
- Verify input data format matches training data

**CORS issues with frontend:**
- Already configured with `Flask-CORS`
- Verify frontend is calling correct API URL
- Check browser console for actual error

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
