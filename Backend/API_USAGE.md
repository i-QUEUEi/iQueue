# iQueue Backend — API usage & frontend examples

Short reference for testing the API (Postman / curl) and a small Vite/React example to call the endpoints and show weekly recommendations.

**Base URL**: https://iqueue-gj4a.onrender.com

**Postman quick setup**
1) Create an Environment variable `baseUrl` with value `https://iqueue-gj4a.onrender.com`.
2) In each request, set URL to `{{baseUrl}}/path`.
3) For POST endpoints, set header `Content-Type: application/json`.

**Endpoints**
- **GET /health** — check service + model/data status
  - Postman: `GET {{baseUrl}}/health`

- **GET /info** — service metadata
  - Postman: `GET {{baseUrl}}/info`

- **POST /predict** — single prediction
  - Postman: `POST {{baseUrl}}/predict`
  - Body (raw JSON):
    ```json
    {
      "date": "2026-05-13",
      "hour": 10,
      "day_of_week": "Wednesday",
      "queue_length_at_arrival": 5
    }
    ```

- **POST /batch-predict** — multiple predictions (useful for weekly UI)
  - Postman: `POST {{baseUrl}}/batch-predict`
  - Body (raw JSON):
    ```json
    {
      "predictions": [
        {"date":"2026-05-13","hour":9,"day_of_week":"Wednesday","queue_length_at_arrival":3},
        {"date":"2026-05-14","hour":14,"day_of_week":"Thursday","queue_length_at_arrival":6}
      ]
    }
    ```

- **GET /api/model-performance** — charts + cards from training outputs
  - Postman: `GET {{baseUrl}}/api/model-performance`

- **GET /api/feature-importance** — feature importances + insights
  - Postman: `GET {{baseUrl}}/api/feature-importance`

- **GET /api/historical-analytics** — daily, hourly, heatmap, insights
  - Postman: `GET {{baseUrl}}/api/historical-analytics`

- **GET /api/predictive-analytics** — model-based time-slot predictions
  - Postman: `GET {{baseUrl}}/api/predictive-analytics`

- **GET /api/dataset-summary** — dataset stats
  - Postman: `GET {{baseUrl}}/api/dataset-summary`

- **GET /api/metrics** — raw metrics.txt + parsed feature importance
  - Postman: `GET {{baseUrl}}/api/metrics`

**Copy-ready endpoint list (Postman)**
```
GET https://iqueue-gj4a.onrender.com/health
GET https://iqueue-gj4a.onrender.com/info
POST https://iqueue-gj4a.onrender.com/predict
POST https://iqueue-gj4a.onrender.com/batch-predict
GET https://iqueue-gj4a.onrender.com/api/model-performance
GET https://iqueue-gj4a.onrender.com/api/feature-importance
GET https://iqueue-gj4a.onrender.com/api/historical-analytics
GET https://iqueue-gj4a.onrender.com/api/predictive-analytics
GET https://iqueue-gj4a.onrender.com/api/dataset-summary
GET https://iqueue-gj4a.onrender.com/api/metrics
```

Notes:
- If `/health` shows `model_loaded: false` after deploy, check logs for the model download error or confirm `MODEL_URL` is a direct `resolve` link.
- The API uses Flask-CORS; browser requests from your frontend should work. If you see CORS errors in the browser console, test with Postman (it ignores CORS) and inspect server logs.

---

**Vite + React example (weekly recommendation)**

Drop this component into a Vite React project (e.g., `src/components/WeeklyRecommend.jsx`). It creates candidate slots for the next 7 days, calls `/batch-predict`, and shows the top 3 best times.

```javascript
import {useEffect, useState} from 'react';

function formatDate(d){
  return d.toISOString().slice(0,10);
}

function getDayName(d){
  return d.toLocaleDateString(undefined, {weekday: 'long'});
}

export default function WeeklyRecommend(){
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  
  useEffect(()=>{
    async function fetchWeek(){
      setLoading(true);
      const base = new Date();
      const candidates = [];
      // generate 7 days × a few hours per day
      for(let day=0; day<7; day++){
        const d = new Date(base);
        d.setDate(base.getDate()+day);
        [9,11,13,15].forEach(hour=>{
          candidates.push({
            date: formatDate(d),
            hour,
            day_of_week: getDayName(d),
            queue_length_at_arrival: 5
          });
        });
      }

      try{
        const res = await fetch('https://iqueue-api.onrender.com/batch-predict', {
          method: 'POST',
          headers: {'Content-Type':'application/json'},
          body: JSON.stringify({predictions: candidates})
        });
        const json = await res.json();
        const good = (json.results || []).filter(r=>r.status==='success')
          .map(r=>({...r, prediction: r.prediction}))
          .sort((a,b)=>a.prediction - b.prediction)
          .slice(0,3);
        setResults(good);
      }catch(e){
        console.error(e);
        setResults([]);
      }finally{setLoading(false)}
    }
    fetchWeek();
  },[]);

  return (
    <div>
      <h3>Top 3 recommended slots (lowest predicted wait)</h3>
      {loading && <p>Loading...</p>}
      {!loading && results && results.length===0 && <p>No results</p>}
      <ul>
        {results && results.map((r,i)=> (
          <li key={i}>{r.input.date} {r.input.hour}:00 — {r.prediction.toFixed(1)} min</li>
        ))}
      </ul>
    </div>
  );
}
```

Usage in a Vite React app:
- Install and run your Vite app as usual (`npm install`, `npm run dev`).
- Import and render the `WeeklyRecommend` component in your app.

---

If you want, I can also add a small Postman collection JSON you can import directly. 
