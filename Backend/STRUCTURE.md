# Backend Folder Structure

This folder contains the production API that gets deployed to Render.

```
Backend/
├── app.py                          # Main Flask API
├── requirements.txt                # Python dependencies
├── Procfile                        # Render deployment config
├── .env.example                    # Environment variables template
├── RENDER_DEPLOYMENT.md            # Deployment guide
│
├── models/                         # ⭐ Copy your model here
│   └── queue_model.pkl             # (paste your trained model)
│
└── data/                           # ⭐ Copy data files here
    ├── synthetic_lto_cdo_queue_90days.csv
    └── 2026-calendar-with-holidays-portrait-sunday-start-en-ph.csv
```

## Setup Instructions

### 1. Copy Model
```bash
# Copy your trained model to:
Backend/models/queue_model.pkl
```

### 2. Copy Data Files
```bash
# Copy these from root data/ folder:
Backend/data/synthetic_lto_cdo_queue_90days.csv
Backend/data/2026-calendar-with-holidays-portrait-sunday-start-en-ph.csv
```

### 3. Local Testing
```bash
cd Backend
pip install -r requirements.txt
python app.py
```
Visit: `http://localhost:5000/health`

### 4. Deploy to Render
See `RENDER_DEPLOYMENT.md` for step-by-step guide.

---

## Why This Structure?

✅ **Only Backend folder gets deployed** - self-contained, fast deployment  
✅ **Model & data in same folder** - easy to manage  
✅ **Path flexibility** - can still import from parent `src/` for preprocessing  

---

## Important Notes

- ⚠️ Model file must exist before deploying (app.py will warn if missing)
- ⚠️ Data files needed for predictions to work correctly
- ⚠️ Keep `src/preprocess.py` in root (backend imports it)
