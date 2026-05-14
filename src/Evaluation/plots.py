"""Plot generation module — creates 4 visualization charts for model evaluation.

All plots use matplotlib in "Agg" (non-interactive) mode, meaning they
generate image files without opening a GUI window. This is essential for
automated pipelines and server environments.
"""
import matplotlib

# Use non-interactive backend — generates images without opening a window
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_target_distribution(df, plots_dir):
    """Chart 1: Histogram of all waiting times in the dataset.

    Shows how wait times are distributed — are most waits short or long?
    A realistic dataset should be right-skewed (most waits short, few very long).

    Saved as: plots_dir/target_distribution.png
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    # 24 bins divides the range into ~3.5 min buckets
    ax.hist(df["waiting_time_min"], bins=24, color="#1f77b4", alpha=0.85, edgecolor="white")
    ax.set_title("Waiting Time Distribution")
    ax.set_xlabel("Waiting time (minutes)")
    ax.set_ylabel("Count")
    fig.tight_layout()
    fig.savefig(plots_dir / "target_distribution.png", dpi=160)
    plt.close(fig)  # Free memory — don't keep figures in RAM


def plot_day_hour_heatmap(df, plots_dir):
    """Chart 2: Heatmap showing average wait time for each day × hour combination.

    Each cell is colored from yellow (short wait) to red (long wait).
    You should see Monday and Friday rows as redder (busier days),
    and columns 9-11 and 14-15 as redder (peak hours).

    Saved as: plots_dir/day_hour_heatmap.png
    """
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]

    # Create a pivot table: rows=days, columns=hours, values=mean wait time
    heatmap = (
        df.pivot_table(index="day_name", columns="hour", values="waiting_time_min", aggfunc="mean")
        .reindex(day_order)        # Ensure Monday is first, Saturday is last
        .sort_index(axis=1)        # Sort hours left to right (8am → 4pm)
    )

    fig, ax = plt.subplots(figsize=(12, 5.5))
    # imshow() renders the matrix as a color grid
    im = ax.imshow(heatmap.values, aspect="auto", cmap="YlOrRd")  # Yellow-Orange-Red colormap
    ax.set_title("Average Waiting Time by Day and Hour")
    ax.set_yticks(range(len(day_order)))
    ax.set_yticklabels(day_order)
    ax.set_xticks(range(len(heatmap.columns)))
    ax.set_xticklabels([f"{int(hour):02d}:00" for hour in heatmap.columns], rotation=45, ha="right")
    fig.colorbar(im, ax=ax, label="Minutes")  # Color legend showing what each color means
    fig.tight_layout()
    fig.savefig(plots_dir / "day_hour_heatmap.png", dpi=160)
    plt.close(fig)


def plot_model_comparison(results_df, plots_dir):
    """Chart 3: Grouped bar chart comparing all 3 models' MAE scores.

    Each model gets 3 bars side by side:
    - Blue: Random split MAE
    - Orange: Chronological MAE
    - Green: Robust MAE (average of all 3 methods)

    The model with the shortest green bar is the winner.

    Saved as: plots_dir/model_comparison.png
    """
    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(results_df))    # One position per model on the x-axis
    width = 0.24                       # Width of each bar (3 bars × 0.24 = 0.72 total)

    # Three grouped bars per model, offset by width
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
    """Chart 4: Scatter plot of actual vs predicted wait times.

    Each dot is one prediction. The red dashed line is y=x (perfect prediction).
    - Dots near the line = accurate model
    - Dots above the line = model over-predicts
    - Dots below the line = model under-predicts

    Saved as: plots_dir/actual_vs_predicted.png
    """
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    # Each dot: x=actual wait, y=predicted wait
    ax.scatter(y_true, y_pred, alpha=0.7, color="#1f77b4", edgecolor="white", linewidth=0.4)

    # Draw the "perfect prediction" diagonal line
    bounds = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(bounds, bounds, linestyle="--", color="#d62728", linewidth=2)  # Red dashed y=x line

    ax.set_title(f"Actual vs Predicted ({model_name})")
    ax.set_xlabel("Actual waiting time (minutes)")
    ax.set_ylabel("Predicted waiting time (minutes)")
    fig.tight_layout()
    fig.savefig(plots_dir / "actual_vs_predicted.png", dpi=160)
    plt.close(fig)
