---
title: iQueue Backend
emoji: 🚦
colorFrom: red
colorTo: orange
sdk: docker
pinned: false
short_description: iQueue ML-powered queue waiting time prediction API
---

# iQueue Prediction Backend

Flask REST API for queue waiting time predictions using a trained Random Forest ML model.

## Endpoints

| Method | Route | Description |
|--------|-------|-------------|
| GET | `/health` | Health check |
| POST | `/predict` | Single prediction |
| POST | `/batch-predict` | Batch predictions |
| GET | `/api/model-performance` | Model metrics |
| GET | `/api/feature-importance` | Feature importances |
| GET | `/api/historical-analytics` | Historical data analytics |

## Example Request

```bash
curl -X POST https://rhnl-iqueue-backend.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"date": "2026-05-17", "hour": 10, "day_of_week": "Monday", "queue_length_at_arrival": 5}'
```
