# Evaluation package
#
# FOLDER STRUCTURE:
# ─────────────────────────────────────────────────────────────
# data_quality/        ← DATA EVALUATION
#   data_evaluation.py    evaluate_data_quality()
#                         Audits raw CSV: rows, dupes, nulls, impossible values
#
# model_quality/       ← ROBUST EVALUATION
#   model_evaluation.py   evaluate_model(), get_feature_importance()
#   metrics.py            compute_metrics() — MAE, RMSE, R²
#   splits.py             chronological_split() — date-based 80/20
#
# outputs/             ← OUTPUTS
#   plots.py              4 PNG charts → outputs/plots/
#   reporting.py          write_report() → outputs/metrics.txt
#   samples.py            sample_predictions() — 6 sanity-check cases
# ─────────────────────────────────────────────────────────────

from .data_quality.data_evaluation import evaluate_data_quality
from .model_quality.model_evaluation import evaluate_model, get_feature_importance
from .model_quality.metrics import compute_metrics
from .model_quality.splits import chronological_split
from .outputs.plots import (
    plot_actual_vs_predicted,
    plot_day_hour_heatmap,
    plot_model_comparison,
    plot_target_distribution,
)
from .outputs.reporting import write_report
from .outputs.samples import sample_predictions
