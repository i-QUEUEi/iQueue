import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def compute_metrics(y_true, y_pred):
    """Calculate the 3 core performance metrics for a model's predictions.

    Args:
        y_true: Array of actual wait times (ground truth).
        y_pred: Array of predicted wait times from the model.

    Returns:
        Dictionary with:
        - mae:  Mean Absolute Error — average minutes off per prediction.
        - rmse: Root Mean Squared Error — like MAE but punishes big mistakes more.
        - r2:   R-squared — fraction of variance explained (1.0 = perfect, 0 = useless).
    """
    return {
        "mae": mean_absolute_error(y_true, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "r2": r2_score(y_true, y_pred),
    }
