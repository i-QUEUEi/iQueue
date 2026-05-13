# Backend Folder Structure

This folder contains the production API that gets deployed to Render.

```
Backend/
â”œâ”€â”€ app.py                          # Main Flask API
â”œâ”€â”€ requirements.txt                # Python dependencies
â”œâ”€â”€ Procfile                        # Render deployment config
â”œâ”€â”€ .env.example                    # Environment variables template
â”œâ”€â”€ RENDER_DEPLOYMENT.md            # Deployment guide
â”‚
â”œâ”€â”€ models/                         # â­ Copy your model here
â”‚   â””â”€â”€ queue_model.pkl             # (paste your trained model)
â”‚
â””â”€â”€ data/                           # â­ Copy data files here
    â”œâ”€â”€ synthetic_lto_cdo_queue_90days.csv
    â””â”€â”€ 2026-calendar-with-holidays-portrait-sunday-start-en-ph.csv
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

âœ… **Only Backend folder gets deployed** - self-contained, fast deployment  
âœ… **Model & data in same folder** - easy to manage  
âœ… **Path flexibility** - can still import from parent `src/` for preprocessing  

---

## Important Notes

- âš ï¸ Model file must exist before deploying (app.py will warn if missing)
- âš ï¸ Data files needed for predictions to work correctly
- âš ï¸ Keep `src/Preprocessing/preprocess.py` in root (backend imports it)


