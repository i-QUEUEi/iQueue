import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"

print("=" * 60)
print("iQueue System Starting...")
print("=" * 60)

# ── Step 1: Preprocessing ─────────────────────────────────────
# Runs the preprocessing pipeline and saves the cleaned +
# feature-engineered DataFrame to data/Preprocessed/preprocessed_data.csv.
# Training (Step 2) will load that file directly, skipping re-preprocessing.
print("\n[1/3] Running preprocessing...")
subprocess.run(
    [sys.executable, "-m", "Preprocessing.preprocess"],
    cwd=str(SRC_DIR),   # must run from src/ so package imports resolve
    check=True,
)

# ── Step 2: Training ──────────────────────────────────────────
# Loads the preprocessed CSV, trains all 3 models, picks the winner,
# saves the model to models/queue_model.pkl, and generates output charts.
print("\n[2/3] Training model...")
subprocess.run(
    [sys.executable, "src/model_implementation/train_model.py"],
    cwd=str(ROOT_DIR),
    check=True,
)

# ── Step 3: Prediction ────────────────────────────────────────
print("\n[3/3] Running prediction...")
subprocess.run(
    [sys.executable, "src/Prediction/predict.py"],
    cwd=str(ROOT_DIR),
    check=True,
)

print("\n" + "=" * 60)
print("Pipeline complete.")
print("=" * 60)