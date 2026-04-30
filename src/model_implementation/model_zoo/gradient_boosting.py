from sklearn.ensemble import GradientBoostingRegressor


def build_gradient_boosting(random_state, params=None):
    default_params = {
        "n_estimators": 250,
        "learning_rate": 0.05,
        "max_depth": 3,
        "subsample": 0.9,
        "random_state": random_state,
    }
    if params:
        default_params.update(params)
    return GradientBoostingRegressor(**default_params)