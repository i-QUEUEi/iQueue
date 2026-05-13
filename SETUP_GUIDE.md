# 🚀 iQueue - Running the Full System

This guide explains how to run both the backend API and frontend application to display real data from your queue dataset.

## 📋 Prerequisites

- Python 3.8+
- Node.js 16+
- npm or yarn
- Pandas, Flask, Flask-CORS installed (see requirements.txt)

## 🔧 Installation

### 1. Backend Setup

Install Python dependencies:

```bash
pip install -r requirements.txt
```

### 2. Frontend Setup

Install Node dependencies:

```bash
cd iQueue
npm install
```

Create a `.env.local` file in the `iQueue` directory:

```bash
cp iQueue/.env.example iQueue/.env.local
```

## 🎯 Running the System

You'll need **two terminal windows**:

### Terminal 1: Start the Backend API

```bash
python api.py
```

Expected output:
```
✅ iQueue API loaded
📊 Dataset: XXXX records from 2026-01-01 to 2026-03-31
🚀 Starting Flask API server on http://localhost:5000
```

The API server will be running on `http://localhost:5000`

**Available endpoints:**
- `GET /api/health` - Health check
- `GET /api/model-performance` - Model performance metrics
- `GET /api/feature-importance` - Feature importance data
- `GET /api/historical-analytics` - Historical queue analytics
- `GET /api/predictive-analytics` - Predictive analytics and forecasts
- `GET /api/dataset-summary` - Dataset statistics

### Terminal 2: Start the Frontend Dev Server

```bash
cd iQueue
npm run dev
```

Expected output:
```
  VITE v5.x.x  build XXXX

  ➜  Local:   http://localhost:5173/
  ➜  press h to show help
```

Open your browser to `http://localhost:5173/` to see the iQueue dashboard.

## 📊 Data Flow

```
CSV Files (data/*.csv)
    ↓
Backend API (api.py) - Processes & serves data
    ↓
Frontend Components - Fetch & Display real data
    ↓
Dashboard Charts & Analytics
```

### How Data is Used

**ModelPerformanceSection:**
- Fetches actual waiting time statistics from the CSV
- Calculates MAE, RMSE, R² scores
- Generates model comparison data

**FeatureImportanceSection:**
- Returns pre-calculated feature importance rankings
- Provides insights based on queue patterns

**HistoricalAnalyticsSection:**
- Aggregates data by day of week and hour
- Creates heatmaps showing temporal patterns
- Calculates statistics for key insights

**PredictiveAnalyticsSection:**
- Provides predictions for different time slots (morning, afternoon, evening)
- System reliability probabilities
- Confidence scores for predictions

## 🔄 Updating Component Data

All landing page components now fetch live data:

```typescript
// Example from ModelPerformanceSection.tsx
import { fetchModelPerformance } from '../../lib/api';

useEffect(() => {
  const loadData = async () => {
    const data = await fetchModelPerformance();
    setChartData(data.chartData);
    // ... set other state
  };
  loadData();
}, []);
```

To add new API endpoints:

1. Add endpoint to `api.py`:
```python
@app.route('/api/my-endpoint', methods=['GET'])
def my_endpoint():
    return jsonify({...})
```

2. Add fetch function to `src/lib/api.ts`:
```typescript
export async function fetchMyEndpoint() {
  const response = await fetch(`${API_BASE_URL}/my-endpoint`);
  return await response.json();
}
```

3. Use in component:
```typescript
import { fetchMyEndpoint } from '../../lib/api';

useEffect(() => {
  const data = await fetchMyEndpoint();
  // Use data...
}, []);
```

## 🛠️ Troubleshooting

### CORS Errors
If you see CORS errors, ensure the backend is running on `localhost:5000` and the frontend `.env.local` has `VITE_API_URL=http://localhost:5000/api`.

### API Not Found
Make sure the backend terminal shows "Starting Flask API server on http://localhost:5000".

### No Data Appearing
1. Check browser console for fetch errors
2. Verify `data/synthetic_lto_cdo_queue_90days.csv` exists
3. Ensure both servers are running

### Build Issues
```bash
cd iQueue
npm run build
```

If build fails, check that all components have proper imports and no hardcoded data remains.

## 📈 Extending the System

The backend can be extended to:

- Generate real-time predictions as new queue data arrives
- Accept user input parameters for custom analytics
- Export reports in various formats
- Integrate with actual LGU queue systems

## 📝 Notes

- The backend uses port `5000` by default
- The frontend dev server uses port `5173`
- CORS is enabled to allow cross-origin requests
- All data is computed on-demand from the CSV file
- No database required (uses CSV as data source)

---

For more details, see:
- Backend: [api.py](./api.py)
- Frontend API utils: [iQueue/src/lib/api.ts](./iQueue/src/lib/api.ts)
- Data source: [data/](./data/)
