"""Linear Regression model utilities."""

from sklearn.linear_model import LinearRegression


def train_linear_regression(X_train, y_train):
    """Fit Linear Regression on the training partition only."""
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model
