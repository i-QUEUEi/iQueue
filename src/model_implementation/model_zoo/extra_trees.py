from sklearn.ensemble import ExtraTreesRegressor


def build_extra_trees(random_state, params=None):
    default_params = {
        "n_estimators": 400,
        "max_depth": 18,
        "min_samples_split": 4,
        "min_samples_leaf": 1,
        "max_features": 0.8,
        "random_state": random_state,
    }
    if params:
        default_params.update(params)
    return ExtraTreesRegressor(**default_params)