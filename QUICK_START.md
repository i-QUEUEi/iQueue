# 🚀 Quick Start - Using Real Data

Your frontend is now using **real data** from the CSV files instead of hardcoded values!

## What Changed

✅ **Backend API Created** (`api.py`)
- Reads data from `data/synthetic_lto_cdo_queue_90days.csv`
- Provides 5 API endpoints with real queue analytics
- Runs on `http://localhost:5000`

✅ **Frontend Updated** - All landing components now fetch live data:
- `ModelPerformanceSection.tsx` - Real model performance metrics
- `FeatureImportanceSection.tsx` - Feature importance rankings
- `HistoricalAnalyticsSection.tsx` - Historical queue patterns
- `PredictiveAnalyticsSection.tsx` - Time-based predictions
- `API utilities` (`src/lib/api.ts`) - Fetch functions for all endpoints

✅ **Environment Configuration** 
- `.env.example` - Template for environment variables
- Frontend connects to backend via `VITE_API_URL`

## How to Run

### Step 1: Install Dependencies
```bash
# Backend dependencies
pip install -r requirements.txt

# Frontend dependencies
cd iQueue
npm install
```

### Step 2: Create Frontend .env File
```bash
cp iQueue/.env.example iQueue/.env.local
```

### Step 3: Run Both Servers (2 Terminal Windows)

**Terminal 1 - Backend API:**
```bash
python api.py
```
Should see:
```
✅ iQueue API loaded
📊 Dataset: XXXX records from 2026-01-01 to 2026-03-31
🚀 Starting Flask API server on http://localhost:5000
```

**Terminal 2 - Frontend Dev Server:**
```bash
cd iQueue
npm run dev
```
Then open `http://localhost:5173` in your browser

## Data Sources

| Component | API Endpoint | Data Source |
|-----------|-------------|-------------|
| Model Performance | `/api/model-performance` | CSV waiting times |
| Feature Importance | `/api/feature-importance` | Static rankings |
| Historical Analytics | `/api/historical-analytics` | Daily/hourly aggregations |
| Predictive Analytics | `/api/predictive-analytics` | Time-based forecasts |
| Dataset Summary | `/api/dataset-summary` | CSV statistics |

## API Endpoints Reference

```
GET /api/health
  → Health check

GET /api/model-performance
  → {comparisonData, performanceMetrics, chartData}

GET /api/feature-importance
  → {importanceData, topInsights}

GET /api/historical-analytics
  → {dailyData, hourlyData, heatmapData, insights}

GET /api/predictive-analytics
  → {predictions, timeSlots, systemReliability}

GET /api/dataset-summary
  → {totalRecords, dateRange, averageWaitTime, ...}
```

## Files Modified/Created

**Backend:**
- ✨ `api.py` - New Flask API server
- 📝 `requirements.txt` - Added Flask, flask-cors

**Frontend:**
- ✨ `iQueue/src/lib/api.ts` - API fetch utilities
- ✨ `iQueue/.env.example` - Environment template
- 🔄 `ModelPerformanceSection.tsx` - Now fetches real data
- 🔄 `FeatureImportanceSection.tsx` - Now fetches real data
- 🔄 `HistoricalAnalyticsSection.tsx` - Now fetches real data
- 🔄 `PredictiveAnalyticsSection.tsx` - Now fetches real data

**Documentation:**
- ✨ `SETUP_GUIDE.md` - Detailed setup instructions

## Next Steps

1. ✅ Run the system following the "How to Run" section above
2. 🔍 Check browser console for any fetch errors
3. 📊 Verify all charts display real data
4. 🎨 Continue styling/enhancing the UI
5. 🚀 Add more data processing in the backend as needed

## Troubleshooting

**Charts show no data?**
- Ensure backend is running: `python api.py`
- Check browser console for fetch errors
- Verify CSV file exists: `data/synthetic_lto_cdo_queue_90days.csv`

**CORS error?**
- Backend must run on `localhost:5000`
- Verify `VITE_API_URL` in `.env.local`

**Build fails?**
- Run `npm install` in iQueue directory
- Check Node version: `node --version` (should be 16+)

---

**Total Records in Dataset:** ~28K queue transactions  
**Date Range:** January 1 - March 31, 2026 (90 days)  
**Key Metrics:** Waiting times, service times, queue lengths, peak hours, day patterns
