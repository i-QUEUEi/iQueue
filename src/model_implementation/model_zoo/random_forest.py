from sklearn.ensemble import RandomForestRegressor


def build_random_forest(random_state, params=None):
    default_params = {
        "n_estimators": 500,
        "max_depth": 15,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "random_state": random_state,
    }
    if params:
        default_params.update(params)
    return RandomForestRegressor(**default_params)