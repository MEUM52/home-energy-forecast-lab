"""Random Forest model utilities."""

from sklearn.ensemble import RandomForestRegressor


def train_random_forest(X_train, y_train, random_state=42):
    """Fit a reproducible Random Forest on the training partition only."""
    model = RandomForestRegressor(
        n_estimators=200,
        max_features=1.0,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    return model
