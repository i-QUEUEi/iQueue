from sklearn.ensemble import RandomForestRegressor


def build_random_forest(random_state, params=None):
    """Build a Random Forest model with tuned hyperparameters.

    Formula: ŷ = average of 500 independent decision trees

    Each tree is trained on a random subset of data and features,
    then the final prediction is the average of all 500 trees.
    This "wisdom of the crowd" approach is more robust than any single tree.

    Args:
        random_state: Seed for reproducibility.
        params: Optional dict to override default hyperparameters.

    Returns:
        A configured RandomForestRegressor ready for training.
    """
    default_params = {
        "n_estimators": 500,        # Number of decision trees to build
        "max_depth": 15,            # Maximum depth per tree (prevents memorization)
        "min_samples_split": 5,     # Need at least 5 samples to split a node
        "min_samples_leaf": 2,      # Each leaf must contain at least 2 data points
        "max_features": "sqrt",     # Each tree randomly sees only √16 ≈ 4 features
        "random_state": random_state,  # Ensures reproducible results
    }
    # Allow caller to override any default parameter
    if params:
        default_params.update(params)
    return RandomForestRegressor(**default_params)