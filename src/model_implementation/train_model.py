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
    ensure_output_dirs()

    raw_df = pd.read_csv(DATA_PATH)
    summary = evaluate_data_quality(raw_df)

    df = load_data(DATA_PATH)
    X, y, features = get_features(df)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE)
    chrono_train, chrono_test, time_train_dates, time_test_dates = chronological_split(df, features)

    models = build_model_catalog(RANDOM_STATE)
    results = []
    selected_result = None

    for name, model in models.items():
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

    print("\n📊 MODEL PERFORMANCE")
    for _, row in results_df.iterrows():
        print(
            f"{row['model']}: robust MAE={row['robust_mae']:.2f}, test MAE={row['test_mae']:.2f}, chrono MAE={row['chrono_test_mae']:.2f}"
        )

    baseline_value = y_train.mean()
    baseline_test_pred = np.full(y_test.shape, baseline_value, dtype=float)
    baseline_metrics = compute_metrics(y_test, baseline_test_pred)

    print(f"\n✅ Selected model: {selected_model}")
    print(f"Selected test MAE: {selected_result['test_metrics']['mae']:.2f} minutes")
    print(f"Selected chronological MAE: {selected_result['chrono_metrics']['mae']:.2f} minutes")
    print(f"Baseline MAE: {baseline_metrics['mae']:.2f} minutes")

    joblib.dump(selected_result["model"], MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")

    plot_target_distribution(df, PLOTS_DIR)
    plot_day_hour_heatmap(df, PLOTS_DIR)
    plot_model_comparison(results_df, PLOTS_DIR)
    plot_actual_vs_predicted(y_test.to_numpy(), selected_result["test_pred"], selected_model, PLOTS_DIR)

    results_df.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)
    write_report(summary, results_df, selected_result, baseline_metrics, time_train_dates, time_test_dates, OUTPUTS_DIR)

    sample_predictions(selected_result["model"], features)

    print(f"\n✅ Metrics saved to {OUTPUTS_DIR / 'metrics.txt'}")
    print(f"✅ Visuals saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
