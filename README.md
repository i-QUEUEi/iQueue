# i-QUEUE

Queue waiting-time forecasting CLI for LTO CDO.

## What This Project Uses

- Synthetic queue dataset in [data/synthetic_lto_cdo_queue_90days.csv](data/synthetic_lto_cdo_queue_90days.csv)
- Multi-model benchmark trainer with robust evaluation in [src/model_implementation/train_model.py](src/model_implementation/train_model.py)
- Monte Carlo uncertainty simulation in [src/Prediction/predict.py](src/Prediction/predict.py)

## Prerequisites

- Python 3.10+
- Git

## Team Setup (First Time)

### 1. Clone the repository

```bash
git clone https://github.com/i-QUEUEi/i-QUEUE.git
cd i-QUEUE
```

### 2. Create and activate a virtual environment

Windows (PowerShell):

```powershell
python -m venv venv
venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

## Run the Project

### Option A: Full run (recommended for first run)

This trains the model, then launches the CLI forecast tool.

```bash
python main.py
```

### Option B: Train only

```bash
python src/model_implementation/train_model.py
```

### Option C: Predict only (if model already exists)

```bash
python src/Prediction/predict.py
```

## Optional: Regenerate Synthetic Data

Only do this if you intentionally want a new generated dataset.

```bash
cd data
python Data_.py
cd ..
python src/model_implementation/train_model.py
```

## Generated Outputs

- Model file: [models/queue_model.pkl](models/queue_model.pkl)
- Metrics report: [outputs/metrics.txt](outputs/metrics.txt)
- Model comparison table: [outputs/model_comparison.csv](outputs/model_comparison.csv)
- Plots: [outputs/plots](outputs/plots)

## Daily Team Workflow

```bash
git pull
venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/Prediction/predict.py
```

Retrain only when:

- data generation changed in [data/Data_.py](data/Data_.py)
- feature engineering changed in [src/Preprocessing/preprocess.py](src/Preprocessing/preprocess.py)
- model settings changed in [src/model_implementation/train_model.py](src/model_implementation/train_model.py)

## CLI Notes

- Input date format: `YYYY-MM-DD` or `today`
- Working days supported: Monday to Saturday
- Sunday input is rejected by design

## Troubleshooting

- `ModuleNotFoundError`:
	- Activate venv and reinstall requirements.
- `FileNotFoundError: models/queue_model.pkl`:
	- Run `python src/model_implementation/train_model.py` first.
- Imports unresolved in VS Code:
	- Select the same interpreter as the project venv.

