from .gradient_boosting import build_gradient_boosting
from .linear_regression import build_linear_regression
from .random_forest import build_random_forest


def build_model_catalog(random_state):
    return {
        "LinearRegression": build_linear_regression(),
        "RandomForest": build_random_forest(random_state),
        "GradientBoosting": build_gradient_boosting(random_state),
    }