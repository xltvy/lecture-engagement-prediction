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


# ---------------------------------------------------------------------------
# Advanced models
# ---------------------------------------------------------------------------

def train_random_forest(
    X,
    y,
    hyperparams: dict = None,
    tune: bool = False,
    n_iter: int = 20,
    random_state: int = 42,
):
    """
    Train a Random Forest classifier, optionally with hyperparameter tuning
    via RandomizedSearchCV (5-fold CV, scoring=matthews_corrcoef).

    Parameters
    ----------
    X : array-like
    y : array-like
        Binary labels.
    hyperparams : dict or None
        Manual hyperparameter overrides (used when tune=False).
    tune : bool
        If True, run a randomised grid search over RF_PARAM_GRID.
    n_iter : int
        Number of parameter settings sampled when tune=True.
    random_state : int

    Returns
    -------
    RandomForestClassifier
        Best fitted model.
    """
    base_params = {**RANDOM_FOREST_DEFAULTS, "random_state": random_state}
    if hyperparams:
        base_params.update(hyperparams)

    if tune:
        print(f"Random Forest: randomised search ({n_iter} configs, 5-fold CV)...")
        base_estimator = RandomForestClassifier(
            class_weight="balanced", random_state=random_state, n_jobs=-1
        )
        search = RandomizedSearchCV(
            base_estimator,
            RF_PARAM_GRID,
            n_iter=n_iter,
            cv=5,
            scoring="matthews_corrcoef",
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )
        search.fit(X, y)
        model = search.best_estimator_
        print(f"Best params: {search.best_params_}")
    else:
        model = RandomForestClassifier(**base_params)
        model.fit(X, y)
        print("Random Forest trained with fixed hyperparameters.")

    return model


def train_xgboost(
    X,
    y,
    hyperparams: dict = None,
    scale_pos_weight: float = None,
    tune: bool = False,
    n_iter: int = 25,
    random_state: int = 42,
):
    """
    Train an XGBoost classifier, optionally with hyperparameter tuning.

    Parameters
    ----------
    X : array-like
    y : array-like
        Binary labels.
    hyperparams : dict or None
        Manual hyperparameter overrides (used when tune=False).
    scale_pos_weight : float or None
        Ratio of negative to positive samples; set automatically from y
        if None.
    tune : bool
        If True, run a randomised grid search over XGB_PARAM_GRID.
    n_iter : int
    random_state : int

    Returns
    -------
    xgboost.XGBClassifier
        Best fitted model.
    """
    if not XGB_AVAILABLE:
        raise ImportError("xgboost is not installed. Run: pip install xgboost")

    if scale_pos_weight is None:
        neg = np.sum(y == 0)
        pos = np.sum(y == 1)
        scale_pos_weight = neg / pos if pos > 0 else 1.0

    base_params = {
        **XGBOOST_DEFAULTS,
        "scale_pos_weight": scale_pos_weight,
        "random_state": random_state,
    }
    if hyperparams:
        base_params.update(hyperparams)

    if tune:
        print(f"XGBoost: randomised search ({n_iter} configs, 5-fold CV)...")
        base_estimator = xgb.XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            random_state=random_state,
            n_jobs=-1,
            eval_metric="logloss",
        )
        search = RandomizedSearchCV(
            base_estimator,
            XGB_PARAM_GRID,
            n_iter=n_iter,
            cv=5,
            scoring="matthews_corrcoef",
            n_jobs=-1,
            random_state=random_state,
            verbose=1,
        )
        search.fit(X, y)
        model = search.best_estimator_
        print(f"Best params: {search.best_params_}")
    else:
        model = xgb.XGBClassifier(**base_params)
        model.fit(X, y)
        print(f"XGBoost trained (scale_pos_weight={scale_pos_weight:.2f}).")

    return model


def build_voting_ensemble(rf_model, xgb_model):
    """
    Build a soft-voting ensemble combining a Random Forest and an XGBoost
    model.  Both must already be fitted.

    Parameters
    ----------
    rf_model : RandomForestClassifier
    xgb_model : XGBClassifier

    Returns
    -------
    VotingClassifier  (not yet fitted — call .fit() separately)
    """
    ensemble = VotingClassifier(
        estimators=[("rf", rf_model), ("xgb", xgb_model)],
        voting="soft",
        n_jobs=-1,
    )
    return ensemble
