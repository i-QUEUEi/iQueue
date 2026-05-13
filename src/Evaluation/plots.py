import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_target_distribution(df, plots_dir):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(df["waiting_time_min"], bins=24, color="#1f77b4", alpha=0.85, edgecolor="white")
    ax.set_title("Waiting Time Distribution")
    ax.set_xlabel("Waiting time (minutes)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(plots_dir / "target_distribution.png", dpi=160)
    plt.close(fig)


def plot_day_hour_heatmap(df, plots_dir):
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
    fig.savefig(plots_dir / "day_hour_heatmap.png", dpi=160)
    plt.close(fig)


def plot_model_comparison(results_df, plots_dir):
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
    fig.savefig(plots_dir / "model_comparison.png", dpi=160)
    plt.close(fig)


def plot_actual_vs_predicted(y_true, y_pred, model_name, plots_dir):
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.scatter(y_true, y_pred, alpha=0.7, color="#1f77b4", edgecolor="white", linewidth=0.4)
    bounds = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(bounds, bounds, linestyle="--", color="#d62728", linewidth=2)
    ax.set_title(f"Actual vs Predicted ({model_name})")
    ax.set_xlabel("Actual waiting time (minutes)")
    ax.set_ylabel("Predicted waiting time (minutes)")
    fig.tight_layout()
    fig.savefig(plots_dir / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)
