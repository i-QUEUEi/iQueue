from sklearn.ensemble import GradientBoostingRegressor


def build_gradient_boosting(random_state, params=None):
    """Build a Gradient Boosting model with tuned hyperparameters.

    Formula: ŷ = F₀ + η·h₁(x) + η·h₂(x) + ... + η·h₂₅₀(x)

    Trees are built SEQUENTIALLY — each new tree focuses on correcting
    the mistakes of all previous trees. The learning rate (η) controls
    how much each tree is allowed to correct (small = conservative).

    Args:
        random_state: Seed for reproducibility.
        params: Optional dict to override default hyperparameters.

    Returns:
        A configured GradientBoostingRegressor ready for training.
    """
    default_params = {
        "n_estimators": 250,       # Number of sequential correction rounds
        "learning_rate": 0.05,     # Each tree only fixes 5% of remaining error
        "max_depth": 3,            # Deliberately shallow — complexity from combining many
        "subsample": 0.9,          # Each tree trains on random 90% of data (regularization)
        "random_state": random_state,  # Ensures reproducible results
    }
    # Allow caller to override any default parameter
    if params:
        default_params.update(params)
    return GradientBoostingRegressor(**default_params)