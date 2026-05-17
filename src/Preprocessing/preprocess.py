"""Preprocessing convenience hub — re-exports all preprocessing functions.

Can also be run as a standalone script to verify the data pipeline:

    python src/Preprocessing/preprocess.py

This will load the raw CSV, run all feature engineering, print a summary,
and save the cleaned DataFrame to data/Preprocessed/preprocessed_data.csv.
No model is trained.

This module doesn't contain any logic. It imports functions from the other
3 preprocessing files (loader, features, calendar) and re-exports them,
so the rest of the codebase can import everything from one place:

    from Preprocessing.preprocess import load_data, get_features

Instead of:

    from Preprocessing.loader import load_data
    from Preprocessing.features import get_features
    from Preprocessing.calendar import load_ph_holiday_month_days
"""

import sys
from pathlib import Path

# ── When run directly, ensure src/ is on sys.path so absolute imports work ──
# This must happen BEFORE the try/except import block below.
# When imported as a package the relative imports take precedence and this
# block has no harmful side-effect.
_SRC_DIR = Path(__file__).resolve().parent.parent   # .../src
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

try:
    # Package import path: `from Preprocessing.preprocess import ...`
    from .calendar import load_ph_holiday_month_days
    from .features import FEATURES, build_feature_dataframe, get_features
    from .loader import load_data
except ImportError:
    # Direct-run path: `python src/Preprocessing/preprocess.py`
    # sys.path was patched above so these absolute imports resolve correctly.
    from Preprocessing.calendar import load_ph_holiday_month_days
    from Preprocessing.features import FEATURES, build_feature_dataframe, get_features
    from Preprocessing.loader import load_data

# Explicit list of what this module exports when someone does `from preprocess import *`
__all__ = [
    "FEATURES",
    "build_feature_dataframe",
    "get_features",
    "load_data",
    "load_ph_holiday_month_days",
]

if __name__ == "__main__":
    """Standalone preprocessing runner.

    Resolves paths relative to the project root (two levels above this file),
    loads the raw CSV, engineers all features, prints a diagnostic summary,
    and saves the preprocessed DataFrame to data/Preprocessed/.
    No model is trained.
    """
    import sys
    from pathlib import Path

    # ── Resolve project root and add src/ to import path ──────────────────────
    # __file__ is  .../src/Preprocessing/preprocess.py
    # parents[1]  →  .../src
    # parents[2]  →  project root  (iQueue/)
    ROOT_DIR = Path(__file__).resolve().parents[2]
    SRC_DIR = ROOT_DIR / "src"
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))

    DATA_PATH = ROOT_DIR / "data" / "synthetic_lto_cdo_queue_90days.csv"
    OUTPUT_DIR = ROOT_DIR / "data" / "Preprocessed"
    OUTPUT_CSV = OUTPUT_DIR / "preprocessed_data.csv"

    print("\n" + "=" * 60)
    print("[iQueue] Preprocessing (standalone)")
    print("=" * 60)
    print(f"   Source  : {DATA_PATH}")
    print(f"   Output  : {OUTPUT_CSV}")

    if not DATA_PATH.exists():
        print(f"\n[ERROR] Data file not found: {DATA_PATH}")
        print("   Make sure the CSV is in the data/ directory.")
        sys.exit(1)

    # ── Run the full preprocessing pipeline ───────────────────────────────────
    df = load_data(DATA_PATH)           # load + clean + engineer features
    X, y, features = get_features(df)   # extract X / y (for summary stats)

    # ── Print summary ─────────────────────────────────────────────────────────
    print("\n[Summary] Preprocessed DataFrame:")
    print(f"   Total rows   : {len(df):,}")
    print(f"   Total columns: {len(df.columns)}")
    print(f"   Feature cols : {len(features)}")
    print(f"   Target (y)   : waiting_time_min")
    print(f"   y -- min: {y.min():.1f}  mean: {y.mean():.1f}  max: {y.max():.1f}")
    print(f"   Missing values after cleaning: {df.isnull().sum().sum()}")

    # ── Save preprocessed DataFrame to CSV ────────────────────────────────────
    # Create the output directory if it doesn't already exist
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)  # write full cleaned + feature DataFrame

    print(f"\n[DONE] Preprocessed data saved -> {OUTPUT_CSV}")
    print("=" * 60)