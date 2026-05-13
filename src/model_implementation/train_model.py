from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from Preprocessing.preprocess import get_features, load_data
from model_implementation import build_model_catalog
from Evaluation.evaluation import evaluate_data_quality, evaluate_model
from Evaluation.metrics import compute_metrics
from Evaluation.plots import (
    plot_actual_vs_predicted,
    plot_day_hour_heatmap,
    plot_model_comparison,
    plot_target_distribution,
)
from Evaluation.reporting import write_report
from Evaluation.samples import sample_predictions
from Evaluation.splits import chronological_split

DATA_PATH = ROOT_DIR / "data" / "synthetic_lto_cdo_queue_90days.csv"
MODEL_PATH = ROOT_DIR / "models" / "queue_model.pkl"
OUTPUTS_DIR = ROOT_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

RANDOM_STATE = 42


def ensure_output_dirs():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def main():
    print("\n" + "=" * 70)
    print("🤖 iQueue — ML Training Pipeline")
    print("=" * 70)

    ensure_output_dirs()

    print("\n📂 [Step 1/6] Checking data quality...")
    raw_df = pd.read_csv(DATA_PATH)
    summary = evaluate_data_quality(raw_df)
    print(f"   Rows: {summary['rows']:,}  |  Duplicates: {summary['duplicate_rows']}  |  Missing: {summary['missing_cells']}")
    print(f"   Wait time — Mean: {summary['target_mean']:.1f} min, Std: {summary['target_std']:.1f} min, Max: {summary['target_max']:.1f} min")

    print("\n🔧 [Step 2/6] Loading and preprocessing data...")
    df = load_data(DATA_PATH)
    X, y, features = get_features(df)
    print(f"   Training features: {len(features)} columns")
    print(f"   Target column: waiting_time_min")

    print("\n✂️  [Step 3/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    chrono_train, chrono_test, time_train_dates, time_test_dates = chronological_split(df, features)
    print(f"   Random split  — Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")
    print(f"   Chrono split  — Train: {time_train_dates} dates | Test: {time_test_dates} dates")

    print("\n🏋️  [Step 4/6] Training and evaluating all models...")
    print("   (Each model runs: random split + chronological split + 5-fold CV)")
    models = build_model_catalog(RANDOM_STATE)
    results = []
    selected_result = None

    for name, model in models.items():
        print(f"\n   ┌─ 🔄 Now training: {name}...")
        result = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            chrono_train,
            chrono_test,
            features,
            RANDOM_STATE,
        )
        print(f"   │   Random split  → MAE: {result['test_metrics']['mae']:.2f} min  |  R²: {result['test_metrics']['r2']:.4f}")
        print(f"   │   Chrono split  → MAE: {result['chrono_metrics']['mae']:.2f} min  |  R²: {result['chrono_metrics']['r2']:.4f}")
        print(f"   │   5-Fold CV     → MAE: {result['cv_mae']:.2f} min  |  R²: {result['cv_r2']:.4f}")
        print(f"   │")
        # --- Model-specific internal computation values ---
        fitted = result["model"]
        if name == "LinearRegression":
            print(f"   │   📐 Formula: ŷ = β₀ + β₁x₁ + β₂x₂ + ... + β₁₆x₁₆")
            print(f"   │   Intercept (β₀)  : {fitted.intercept_:.4f}")
            coef_pairs = sorted(zip(features, fitted.coef_), key=lambda x: abs(x[1]), reverse=True)
            print(f"   │   Top coefficients (how much each feature shifts wait time):")
            for feat, coef in coef_pairs[:5]:
                direction = "↑" if coef > 0 else "↓"
                print(f"   │     {feat:<30} β = {coef:+.4f}  {direction}")
        elif name == "RandomForest":
            print(f"   │   🌲 Formula: ŷ = average of {fitted.n_estimators} decision trees")
            depths = [t.get_depth() for t in fitted.estimators_]
            print(f"   │   Trees        : {fitted.n_estimators}  |  Max depth: {fitted.max_depth}")
            print(f"   │   Actual depths: min={min(depths)}, avg={sum(depths)/len(depths):.1f}, max={max(depths)}")
            print(f"   │   Top feature importances (how much each feature reduces prediction error):")
            fi = result["feature_importance"].head(5)
            for _, row in fi.iterrows():
                bar = "█" * int(row["importance"] * 40)
                print(f"   │     {row['feature']:<30} {row['importance']:.4f}  {bar}")
        elif name == "GradientBoosting":
            print(f"   │   🚀 Formula: ŷ = F₀ + η·h₁(x) + η·h₂(x) + ... + η·h₂₅₀(x)")
            print(f"   │   Trees (stages): {fitted.n_estimators_}  |  Learning rate (η): {fitted.learning_rate}")
            print(f"   │   Max depth per tree: {fitted.max_depth}  |  Subsample: {fitted.subsample}")
            print(f"   │   Top feature importances:")
            fi = result["feature_importance"].head(5)
            for _, row in fi.iterrows():
                bar = "█" * int(row["importance"] * 40)
                print(f"   │     {row['feature']:<30} {row['importance']:.4f}  {bar}")
        print(f"   └─ ✅ Robust MAE: {result['robust_mae']:.2f} min")
        results.append(
            {
                "model": name,
                "train_mae": result["train_metrics"]["mae"],
                "train_rmse": result["train_metrics"]["rmse"],
                "train_r2": result["train_metrics"]["r2"],
                "test_mae": result["test_metrics"]["mae"],
                "test_rmse": result["test_metrics"]["rmse"],
                "test_r2": result["test_metrics"]["r2"],
                "chrono_test_mae": result["chrono_metrics"]["mae"],
                "chrono_test_rmse": result["chrono_metrics"]["rmse"],
                "chrono_test_r2": result["chrono_metrics"]["r2"],
                "cv_mae": result["cv_mae"],
                "cv_rmse": result["cv_rmse"],
                "cv_r2": result["cv_r2"],
                "robust_mae": result["robust_mae"],
            }
        )

        if selected_result is None or result["robust_mae"] < selected_result["robust_mae"]:
            selected_result = result

    results_df = pd.DataFrame(results).sort_values("robust_mae")
    selected_model = selected_result["name"]

    print("\n" + "=" * 70)
    print("📊 [Step 5/6] MODEL BENCHMARK RESULTS (ranked by Robust MAE)")
    print("=" * 70)
    print(f"   {'Model':<22} {'Robust MAE':>12} {'Test MAE':>10} {'Chrono MAE':>12} {'Test R²':>9}")
    print("   " + "-" * 66)
    for _, row in results_df.iterrows():
        marker = " ← WINNER" if row["model"] == selected_model else ""
        print(
            f"   {row['model']:<22} {row['robust_mae']:>10.2f}   {row['test_mae']:>8.2f}   {row['chrono_test_mae']:>10.2f}   {row['test_r2']:>7.4f}{marker}"
        )

    baseline_value = y_train.mean()
    baseline_test_pred = np.full(y_test.shape, baseline_value, dtype=float)
    baseline_metrics = compute_metrics(y_test, baseline_test_pred)
    print(f"\n   Baseline (always guess avg): MAE = {baseline_metrics['mae']:.2f} min  |  R² = {baseline_metrics['r2']:.4f}")
    print(f"   Best model ({selected_model}) beats baseline by {baseline_metrics['mae'] - selected_result['test_metrics']['mae']:.2f} min MAE")

    print("\n" + "=" * 70)
    print(f"✅ Selected: {selected_model}")
    print(f"   Test MAE        : {selected_result['test_metrics']['mae']:.2f} min")
    print(f"   Chrono MAE      : {selected_result['chrono_metrics']['mae']:.2f} min")
    print(f"   5-Fold CV MAE   : {selected_result['cv_mae']:.2f} min")
    print(f"   Test R²         : {selected_result['test_metrics']['r2']:.4f}")
    print(f"   P90 error       : {selected_result['p90_abs_error']:.2f} min")
    print(f"   Max error       : {selected_result['max_abs_error']:.2f} min")
    print("=" * 70)

    joblib.dump(selected_result["model"], MODEL_PATH)
    print(f"\n💾 Model saved → {MODEL_PATH}")

    print("\n📈 [Step 6/6] Generating outputs...")
    plot_target_distribution(df, PLOTS_DIR)
    print("   ✅ target_distribution.png")
    plot_day_hour_heatmap(df, PLOTS_DIR)
    print("   ✅ day_hour_heatmap.png")
    plot_model_comparison(results_df, PLOTS_DIR)
    print("   ✅ model_comparison.png")
    plot_actual_vs_predicted(y_test.to_numpy(), selected_result["test_pred"], selected_model, PLOTS_DIR)
    print("   ✅ actual_vs_predicted.png")

    results_df.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)
    write_report(summary, results_df, selected_result, baseline_metrics, time_train_dates, time_test_dates, OUTPUTS_DIR)
    print(f"   ✅ metrics.txt and model_comparison.csv saved")

    sample_predictions(selected_result["model"], features)

    print("\n" + "=" * 70)
    print("🎉 Training complete! All outputs saved.")
    print(f"   Model   : {MODEL_PATH}")
    print(f"   Report  : {OUTPUTS_DIR / 'metrics.txt'}")
    print(f"   Charts  : {PLOTS_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
