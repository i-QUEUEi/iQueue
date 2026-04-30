from pathlib import Path
import sys

import joblib
import matplotlib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_validate, train_test_split


matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from preprocess import get_features, load_data
from model_implementation import build_model_catalog


DATA_PATH = ROOT_DIR / "data" / "synthetic_lto_cdo_queue_90days.csv"
MODEL_PATH = ROOT_DIR / "models" / "queue_model.pkl"
OUTPUTS_DIR = ROOT_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"

RANDOM_STATE = 42

def compute_metrics(y_true, y_pred):
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }


def ensure_output_dirs():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)


def evaluate_data_quality(raw_df):
    target = raw_df["waiting_time_min"]
    queue = raw_df["queue_length_at_arrival"]

    return {
        "rows": int(len(raw_df)),
        "columns": int(raw_df.shape[1]),
        "duplicate_rows": int(raw_df.duplicated().sum()),
        "missing_cells": int(raw_df.isna().sum().sum()),
        "negative_waiting_rows": int((target < 0).sum()),
        "negative_queue_rows": int((queue < 0).sum()),
        "target_mean": float(target.mean()),
        "target_median": float(target.median()),
        "target_std": float(target.std()),
        "target_min": float(target.min()),
        "target_p10": float(target.quantile(0.10)),
        "target_p90": float(target.quantile(0.90)),
        "target_max": float(target.max()),
    }


def plot_target_distribution(df):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["waiting_time_min"], bins=24, color="#1f77b4", alpha=0.85, edgecolor="white")
    ax.set_title("Waiting Time Distribution")
    ax.set_xlabel("Waiting time (minutes)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "target_distribution.png", dpi=160)
    plt.close(fig)


def plot_day_hour_heatmap(df):
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    heatmap = (
        df.pivot_table(index="day_name", columns="hour", values="waiting_time_min", aggfunc="mean")
        .reindex(day_order)
        .sort_index(axis=1)
    )

    fig, ax = plt.subplots(figsize=(12, 5.5))
    im = ax.imshow(heatmap.values, aspect="auto", cmap="YlOrRd")
    ax.set_title("Average Waiting Time by Day and Hour")
    ax.set_yticks(range(len(day_order)))
    ax.set_yticklabels(day_order)
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels([f"{int(hour):02d}:00" for hour in heatmap.columns], rotation=45, ha="right")
    fig.colorbar(im, ax=ax, label="Minutes")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "day_hour_heatmap.png", dpi=160)
    plt.close(fig)


def plot_model_comparison(results_df):
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(results_df))
    width = 0.24

    ax.bar(x - width, results_df["test_mae"], width=width, label="Random split MAE", color="#4c78a8")
    ax.bar(x, results_df["chrono_test_mae"], width=width, label="Chronological MAE", color="#f58518")
    ax.bar(x + width, results_df["robust_mae"], width=width, label="Robust MAE", color="#54a24b")

    ax.set_title("Model Comparison")
    ax.set_ylabel("MAE (minutes)")
    ax.set_xticks(x)
    ax.set_xticklabels(results_df["model"], rotation=15, ha="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "model_comparison.png", dpi=160)
    plt.close(fig)


def plot_actual_vs_predicted(y_true, y_pred, model_name):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(y_true, y_pred, alpha=0.7, color="#1f77b4", edgecolor="white", linewidth=0.4)
    bounds = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(bounds, bounds, linestyle="--", color="#d62728", linewidth=2)
    ax.set_title(f"Actual vs Predicted ({model_name})")
    ax.set_xlabel("Actual waiting time (minutes)")
    ax.set_ylabel("Predicted waiting time (minutes)")
    fig.tight_layout()
    fig.savefig(PLOTS_DIR / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)


def get_feature_importance(model, features, X_reference, y_reference):
    if hasattr(model, "feature_importances_"):
        values = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        values = np.abs(np.asarray(model.coef_, dtype=float)).ravel()
    else:
        importance = permutation_importance(
            model,
            X_reference,
            y_reference,
            n_repeats=10,
            random_state=RANDOM_STATE,
            scoring="neg_mean_absolute_error",
        )
        values = importance.importances_mean

    importance_df = pd.DataFrame({"feature": features, "importance": values})
    total = importance_df["importance"].sum()
    if total > 0:
        importance_df["importance"] = importance_df["importance"] / total
    return importance_df.sort_values("importance", ascending=False)


def evaluate_model(name, model, X_train, X_test, y_train, y_test, chrono_train, chrono_test, features):
    fitted_model = clone(model)
    fitted_model.fit(X_train, y_train)

    train_pred = fitted_model.predict(X_train)
    test_pred = fitted_model.predict(X_test)

    chrono_model = clone(model)
    chrono_model.fit(chrono_train[0], chrono_train[1])
    chrono_pred = chrono_model.predict(chrono_test[0])

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        clone(model),
        X_train,
        y_train,
        cv=cv,
        scoring={
            "mae": "neg_mean_absolute_error",
            "rmse": "neg_root_mean_squared_error",
            "r2": "r2",
        },
    )

    train_metrics = compute_metrics(y_train, train_pred)
    test_metrics = compute_metrics(y_test, test_pred)
    chrono_metrics = compute_metrics(chrono_test[1], chrono_pred)
    cv_mae = float(-cv_results["test_mae"].mean())
    cv_rmse = float(-cv_results["test_rmse"].mean())
    cv_r2 = float(cv_results["test_r2"].mean())

    robust_mae = float(np.mean([test_metrics["mae"], chrono_metrics["mae"], cv_mae]))

    abs_errors = np.abs(y_test.to_numpy() - test_pred)
    p90_abs_error = float(np.percentile(abs_errors, 90))
    p95_abs_error = float(np.percentile(abs_errors, 95))
    max_abs_error = float(np.max(abs_errors))

    eval_df = X_test.copy().reset_index(drop=True)
    eval_df["actual_wait"] = y_test.reset_index(drop=True)
    eval_df["pred_wait"] = test_pred
    eval_df["abs_error"] = np.abs(eval_df["actual_wait"] - eval_df["pred_wait"])

    day_name_map = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
    }
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    eval_df["day_name"] = eval_df["day_of_week"].map(day_name_map)
    day_error = eval_df.groupby("day_name")["abs_error"].agg(["mean", "median", "max", "count"]).reindex(day_order).dropna()
    hour_error = eval_df.groupby("hour")["abs_error"].agg(["mean", "median", "max", "count"]).sort_index()
    peak_day_error = eval_df.groupby("is_peak_day")["abs_error"].mean()
    peak_hour_error = eval_df.groupby("is_peak_hour")["abs_error"].mean()

    feature_importance = get_feature_importance(fitted_model, features, X_test, y_test)

    return {
        "name": name,
        "model": fitted_model,
        "train_pred": train_pred,
        "test_pred": test_pred,
        "chrono_pred": chrono_pred,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "chrono_metrics": chrono_metrics,
        "cv_mae": cv_mae,
        "cv_rmse": cv_rmse,
        "cv_r2": cv_r2,
        "robust_mae": robust_mae,
        "baseline_value": float(np.mean(y_train)),
        "p90_abs_error": p90_abs_error,
        "p95_abs_error": p95_abs_error,
        "max_abs_error": max_abs_error,
        "day_error": day_error,
        "hour_error": hour_error,
        "peak_day_error": peak_day_error,
        "peak_hour_error": peak_hour_error,
        "feature_importance": feature_importance,
        "eval_df": eval_df,
        "chrono_train_size": int(len(chrono_train[0])),
        "chrono_test_size": int(len(chrono_test[0])),
    }


def chronological_split(df, features):
    df_time = df.sort_values("date").copy()
    normalized_dates = df_time["date"].dt.normalize()
    unique_dates = np.sort(normalized_dates.unique())

    split_idx = int(len(unique_dates) * 0.8)
    split_idx = min(max(split_idx, 1), len(unique_dates) - 1)

    time_train_dates = unique_dates[:split_idx]
    time_test_dates = unique_dates[split_idx:]

    time_train_df = df_time[normalized_dates.isin(time_train_dates)]
    time_test_df = df_time[normalized_dates.isin(time_test_dates)]

    X_time_train = time_train_df[features]
    y_time_train = time_train_df["waiting_time_min"]
    X_time_test = time_test_df[features]
    y_time_test = time_test_df["waiting_time_min"]

    return (X_time_train, y_time_train), (X_time_test, y_time_test), len(time_train_dates), len(time_test_dates)


def sample_predictions(model, features):
    print("\n🎯 SAMPLE PREDICTIONS (Using actual patterns):")
    test_cases = [
        {"name": "Monday 9am (Should be 55min)", "hour": 9, "day_of_week": 0, "is_peak_day": 1, "is_peak_hour": 1, "queue": 25, "lag_wait": 25},
        {"name": "Monday 10am (Should be 70min)", "hour": 10, "day_of_week": 0, "is_peak_day": 1, "is_peak_hour": 1, "queue": 35, "lag_wait": 35},
        {"name": "Monday 8am (Should be 25min)", "hour": 8, "day_of_week": 0, "is_peak_day": 1, "is_peak_hour": 0, "queue": 8, "lag_wait": 15},
        {"name": "Wednesday 9am (Should be 19min)", "hour": 9, "day_of_week": 2, "is_peak_day": 0, "is_peak_hour": 1, "queue": 10, "lag_wait": 10},
        {"name": "Wednesday 10am (Should be 25min)", "hour": 10, "day_of_week": 2, "is_peak_day": 0, "is_peak_hour": 1, "queue": 12, "lag_wait": 12},
        {"name": "Wednesday 8am (Should be 9min)", "hour": 8, "day_of_week": 2, "is_peak_day": 0, "is_peak_hour": 0, "queue": 4, "lag_wait": 8},
    ]

    for case in test_cases:
        test_X = pd.DataFrame(
            [[
                case["hour"],
                case["day_of_week"],
                2,
                case["is_peak_day"],
                case["queue"],
                35,
                0,
                case["is_peak_hour"],
                max(2, case["queue"] - 3),
                case["lag_wait"],
            ]],
            columns=features,
        )
        pred = model.predict(test_X)[0]
        print(f"   {case['name']}: {pred:.1f} min")


def write_report(summary, results_df, selected_result, baseline_metrics, time_train_dates, time_test_dates):
    report_path = OUTPUTS_DIR / "metrics.txt"
    selected_name = selected_result["name"]
    model_rows = results_df.copy()
    model_rows["selected"] = model_rows["model"].eq(selected_name)

    with report_path.open("w", encoding="utf-8") as f:
        f.write("DATA EVALUATION\n")
        f.write(f"Rows: {summary['rows']}\n")
        f.write(f"Columns: {summary['columns']}\n")
        f.write(f"Duplicate Rows: {summary['duplicate_rows']}\n")
        f.write(f"Missing Cells: {summary['missing_cells']}\n")
        f.write(f"Negative Waiting Rows: {summary['negative_waiting_rows']}\n")
        f.write(f"Negative Queue Rows: {summary['negative_queue_rows']}\n")
        f.write(f"Target Mean: {summary['target_mean']:.2f}\n")
        f.write(f"Target Median: {summary['target_median']:.2f}\n")
        f.write(f"Target Std: {summary['target_std']:.2f}\n")
        f.write(f"Target Min/Max: {summary['target_min']:.2f} / {summary['target_max']:.2f}\n")
        f.write(f"Target P10/P90: {summary['target_p10']:.2f} / {summary['target_p90']:.2f}\n")

        f.write("\nMODEL BENCHMARK\n")
        for _, row in model_rows.iterrows():
            marker = " [selected]" if bool(row["selected"]) else ""
            f.write(f"{row['model']}{marker}\n")
            f.write(f"  Train MAE: {row['train_mae']:.2f}\n")
            f.write(f"  Test MAE: {row['test_mae']:.2f}\n")
            f.write(f"  Chronological MAE: {row['chrono_test_mae']:.2f}\n")
            f.write(f"  CV MAE: {row['cv_mae']:.2f}\n")
            f.write(f"  Robust MAE: {row['robust_mae']:.2f}\n")
            f.write(f"  Test R2: {row['test_r2']:.4f}\n")
            f.write(f"  Chronological R2: {row['chrono_test_r2']:.4f}\n")

        f.write("\nBASELINE COMPARISON\n")
        f.write(f"Mean Predictor Test MAE: {baseline_metrics['mae']:.2f}\n")
        f.write(f"Mean Predictor Test RMSE: {baseline_metrics['rmse']:.2f}\n")
        f.write(f"Mean Predictor Test R2: {baseline_metrics['r2']:.4f}\n")

        f.write("\nROBUST EVALUATION\n")
        f.write(f"Random Split MAE: {selected_result['test_metrics']['mae']:.2f}\n")
        f.write(f"Random Split RMSE: {selected_result['test_metrics']['rmse']:.2f}\n")
        f.write(f"Random Split R2: {selected_result['test_metrics']['r2']:.4f}\n")
        f.write(f"Chronological Train Dates: {time_train_dates}\n")
        f.write(f"Chronological Test Dates: {time_test_dates}\n")
        f.write(f"Chronological Test MAE: {selected_result['chrono_metrics']['mae']:.2f}\n")
        f.write(f"Chronological Test RMSE: {selected_result['chrono_metrics']['rmse']:.2f}\n")
        f.write(f"Chronological Test R2: {selected_result['chrono_metrics']['r2']:.4f}\n")
        f.write(f"P90 Absolute Error: {selected_result['p90_abs_error']:.2f}\n")
        f.write(f"P95 Absolute Error: {selected_result['p95_abs_error']:.2f}\n")
        f.write(f"Max Absolute Error: {selected_result['max_abs_error']:.2f}\n")

        f.write("\nSEGMENT ERROR CHECKS\n")
        f.write(f"Peak Day MAE (Mon/Fri): {selected_result['peak_day_error'].get(1, np.nan):.2f}\n")
        f.write(f"Non-Peak Day MAE: {selected_result['peak_day_error'].get(0, np.nan):.2f}\n")
        f.write(f"Peak Hour MAE: {selected_result['peak_hour_error'].get(1, np.nan):.2f}\n")
        f.write(f"Non-Peak Hour MAE: {selected_result['peak_hour_error'].get(0, np.nan):.2f}\n")

        f.write("\nWHY THESE MODELS\n")
        f.write("LinearRegression is the simplest baseline and shows whether the pattern is close to linear.\n")
        f.write("RandomForest and ExtraTrees handle nonlinear interactions, noise, and mixed feature scales without manual scaling.\n")
        f.write("GradientBoosting is included because it can capture residual structure with fewer trees than bagging models.\n")

        f.write("\nWHY NOT OTHER OPTIONS\n")
        f.write("We did not rely on only a mean predictor because it ignores hour/day structure.\n")
        f.write("We did not add heavy preprocessing like scaling or one-hot encoding because the inputs are already numeric and tree models are scale-insensitive.\n")
        f.write("We avoided overly complex models such as neural networks because the dataset is small and the signal is mostly tabular pattern learning.\n")

        f.write("\nPREPROCESSING CHOICES\n")
        f.write("Parsed dates and derived week_of_month to preserve within-month seasonality.\n")
        f.write("Filtered negative waiting times and queue lengths, then dropped missing rows for a clean training set.\n")
        f.write("Kept the engineered lag and peak features because they encode the queue dynamics directly.\n")

        f.write("\nFEATURE IMPORTANCE\n")
        for _, row in selected_result["feature_importance"].iterrows():
            f.write(f"{row['feature']}: {row['importance']:.4f}\n")


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
        result = evaluate_model(name, model, X_train, X_test, y_train, y_test, chrono_train, chrono_test, features)
        results.append({
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
        })

        if selected_result is None or result["robust_mae"] < selected_result["robust_mae"]:
            selected_result = result

    results_df = pd.DataFrame(results).sort_values("robust_mae")
    selected_model = selected_result["name"]

    print("\n📊 MODEL PERFORMANCE")
    for _, row in results_df.iterrows():
        print(f"{row['model']}: robust MAE={row['robust_mae']:.2f}, test MAE={row['test_mae']:.2f}, chrono MAE={row['chrono_test_mae']:.2f}")

    baseline_value = y_train.mean()
    baseline_test_pred = np.full(y_test.shape, baseline_value, dtype=float)
    baseline_metrics = compute_metrics(y_test, baseline_test_pred)

    print(f"\n✅ Selected model: {selected_model}")
    print(f"Selected test MAE: {selected_result['test_metrics']['mae']:.2f} minutes")
    print(f"Selected chronological MAE: {selected_result['chrono_metrics']['mae']:.2f} minutes")
    print(f"Baseline MAE: {baseline_metrics['mae']:.2f} minutes")

    joblib.dump(selected_result["model"], MODEL_PATH)
    print(f"\n✅ Model saved to {MODEL_PATH}")

    plot_target_distribution(df)
    plot_day_hour_heatmap(df)
    plot_model_comparison(results_df)
    plot_actual_vs_predicted(y_test.to_numpy(), selected_result["test_pred"], selected_model)

    results_df.to_csv(OUTPUTS_DIR / "model_comparison.csv", index=False)
    write_report(summary, results_df, selected_result, baseline_metrics, time_train_dates, time_test_dates)

    sample_predictions(selected_result["model"], features)

    print(f"\n✅ Metrics saved to {OUTPUTS_DIR / 'metrics.txt'}")
    print(f"✅ Visuals saved to {PLOTS_DIR}")


if __name__ == "__main__":
    main()