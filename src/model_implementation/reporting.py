import numpy as np


def write_report(summary, results_df, selected_result, baseline_metrics, time_train_dates, time_test_dates, output_dir):
    report_path = output_dir / "metrics.txt"
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
        f.write("RandomForest handles nonlinear interactions, noise, and mixed feature scales without manual scaling.\n")
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
