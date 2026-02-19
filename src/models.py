"""
models.py
---------
Model definitions, baseline configurations, and hyperparameter presets
for the lecture engagement prediction task.

Supported models
----------------
- Ridge regression           (regression task)
- Logistic regression        (binary classification task)
- Random Forest classifier   (improved classification)
- XGBoost classifier         (improved classification)
- Voting Ensemble            (soft-voting combination of RF + XGB)
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import RandomizedSearchCV

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# ---------------------------------------------------------------------------
# Default hyperparameter configurations
# ---------------------------------------------------------------------------

RIDGE_DEFAULTS = {
    "alpha": 1.0,
    "random_state": 42,
}

LOGISTIC_DEFAULTS = {
    "max_iter": 1000,
    "random_state": 42,
    "solver": "lbfgs",
    "C": 1.0,
    "class_weight": "balanced",
}

RANDOM_FOREST_DEFAULTS = {
    "n_estimators": 200,
    "max_depth": 15,
    "min_samples_split": 20,
    "min_samples_leaf": 10,
    "max_features": "sqrt",
    "class_weight": "balanced",
    "random_state": 42,
    "n_jobs": -1,
}

XGBOOST_DEFAULTS = {
    "n_estimators": 200,
    "max_depth": 6,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "gamma": 0.1,
    "random_state": 42,
    "n_jobs": -1,
    "eval_metric": "logloss",
}

# Search spaces for RandomizedSearchCV
RF_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [10, 15, 20, None],
    "min_samples_split": [10, 20, 30],
    "min_samples_leaf": [5, 10, 15],
    "max_features": ["sqrt", "log2"],
}

XGB_PARAM_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.01, 0.05, 0.1, 0.2],
    "subsample": [0.7, 0.8, 0.9, 1.0],
    "colsample_bytree": [0.7, 0.8, 0.9, 1.0],
    "gamma": [0, 0.1, 0.2, 0.3],
}


# ---------------------------------------------------------------------------
# Baseline models
# ---------------------------------------------------------------------------

def train_ridge_model(X, y, hyperparams: dict = None):
    """
    Train a Ridge regression model.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
        Continuous engagement labels in [0, 1].
    hyperparams : dict or None
        Overrides for RIDGE_DEFAULTS.

    Returns
    -------
    sklearn.linear_model.Ridge
        Fitted model.
    """
    params = {**RIDGE_DEFAULTS, **(hyperparams or {})}
    model = Ridge(**params)
    model.fit(X, y)
    print(f"Ridge regression trained (alpha={params['alpha']})")
    return model


def train_logistic_model(X, y, hyperparams: dict = None):
    """
    Train a Logistic regression model for binary engagement classification.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
        Binary labels (0 = low engagement, 1 = high engagement).
    hyperparams : dict or None
        Overrides for LOGISTIC_DEFAULTS.

    Returns
    -------
    sklearn.linear_model.LogisticRegression
        Fitted model.
    """
    params = {**LOGISTIC_DEFAULTS, **(hyperparams or {})}
    model = LogisticRegression(**params)
    model.fit(X, y)
    print(f"Logistic regression trained (C={params['C']}, solver={params['solver']})")
    return model