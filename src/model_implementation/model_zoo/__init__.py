"""Model catalog — assembles all available models into a single dictionary.

Adding a new model is simple:
1. Create a new builder file (e.g., xgboost.py) with a build_xgboost() function
2. Import it here
3. Add one line to build_model_catalog()
No changes needed in the training loop.
"""
from .gradient_boosting import build_gradient_boosting
from .linear_regression import build_linear_regression
from .random_forest import build_random_forest


def build_model_catalog(random_state):
    """Build a dictionary of all models to benchmark.

    The training loop iterates over this dictionary, training and evaluating
    each model identically. The model with the lowest robust_mae wins.

    Args:
        random_state: Seed for reproducibility (passed to models that use randomness).

    Returns:
        Dict of {model_name: untrained_model_object}.
    """
    return {
        "LinearRegression": build_linear_regression(),
        "RandomForest": build_random_forest(random_state),
        "GradientBoosting": build_gradient_boosting(random_state),
    }