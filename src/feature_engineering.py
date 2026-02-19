"""
feature_engineering.py
----------------------
Custom feature creation for the lecture engagement prediction task.

All engineered features are derived from the preprocessed feature set.
Target encoding for most_covered_topic is computed on the training set only
and applied to the test set using pre-computed maps, preventing data leakage.
"""

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Target encoding for high-cardinality topic feature
# ---------------------------------------------------------------------------

def compute_topic_encoding(
    X: pd.DataFrame,
    y: np.ndarray,
    min_samples: int = 5,
    smoothing: float = 0.5,
) -> dict:
    """
    Compute Bayesian-smoothed target encoding for most_covered_topic.

    Each topic is mapped to its mean engagement rate.  Topics with fewer
    than *min_samples* observations are smoothed toward the global mean to
    reduce overfitting (Micci-Barreca, 2001).

    Formula for rare topics:
        encoded = smoothing * topic_mean + (1 - smoothing) * global_mean

    Parameters
    ----------
    X : pd.DataFrame
        Training features, must contain most_covered_topic.
    y : np.ndarray
        Continuous engagement labels aligned with X.
    min_samples : int
        Minimum observations required for a reliable per-topic estimate.
    smoothing : float
        Interpolation weight toward the global mean for rare topics.

    Returns
    -------
    dict with keys:
        - topic_means : {topic_str: float}
        - global_mean : float
    """
    assert "most_covered_topic" in X.columns, \
        "most_covered_topic column not found in X"

    global_mean = float(y.mean())
    topics = X["most_covered_topic"].fillna("unknown")
    topic_means = {}

    for topic in topics.unique():
        mask = topics == topic
        count = mask.sum()
        if count >= min_samples:
            topic_means[topic] = float(y[mask.values].mean())
        else:
            raw_mean = float(y[mask.values].mean()) if count > 0 else global_mean
            topic_means[topic] = smoothing * raw_mean + (1 - smoothing) * global_mean

    print(f"Topic encoding: {len(topic_means)} topics | global mean = {global_mean:.4f}")
    return {"topic_means": topic_means, "global_mean": global_mean}


def apply_topic_encoding(
    X: pd.DataFrame,
    encoding_info: dict,
) -> pd.DataFrame:
    """
    Apply pre-computed topic encoding to a feature DataFrame.

    Unseen topics fall back to the global mean.

    Parameters
    ----------
    X : pd.DataFrame
    encoding_info : dict
        As returned by compute_topic_encoding.

    Returns
    -------
    pd.DataFrame with topic_engagement_score added and most_covered_topic removed.
    """
    df = X.copy()
    topic_means = encoding_info["topic_means"]
    global_mean = encoding_info["global_mean"]

    topics = df["most_covered_topic"].fillna("unknown") if "most_covered_topic" in df.columns else None

    if topics is not None:
        df["topic_engagement_score"] = topics.map(
            lambda t: topic_means.get(t, global_mean)
        )
        df = df.drop("most_covered_topic", axis=1)

    return df


# ---------------------------------------------------------------------------
# Interaction and polynomial features
# ---------------------------------------------------------------------------

def add_topic_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction features between the topic engagement score and video
    length / content volume.

    Rationale:
    - topic_duration_interaction : some topics engage better at certain lengths
      (e.g., short tutorials vs. long deep-dives).
    - topic_content_interaction  : content-density requirements vary by topic.

    Parameters
    ----------
    df : pd.DataFrame  (must contain topic_engagement_score)

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    if "topic_engagement_score" not in df.columns:
        return df

    if "duration" in df.columns:
        df["topic_duration_interaction"] = df["topic_engagement_score"] * df["duration"]

    if "word_count" in df.columns:
        df["topic_content_interaction"] = df["topic_engagement_score"] * df["word_count"]

    return df


def add_linguistic_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add entropy_per_word: vocabulary diversity normalised by text length.

    A transcript with high entropy and few words is highly diverse per word;
    longer transcripts naturally accumulate more variety.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    if "document_entropy" in df.columns and "word_count" in df.columns:
        df["entropy_per_word"] = df["document_entropy"] / (df["word_count"] + 1)
    return df


def add_cognitive_load(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cognitive_load: (1 - easiness) * speaker_speed.

    High values indicate difficult content delivered quickly — a pattern
    likely to increase cognitive overload and reduce engagement.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    if "easiness" in df.columns and "speaker_speed" in df.columns:
        df["cognitive_load"] = (1 - df["easiness"]) * df["speaker_speed"]
    return df


def add_polynomial_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add squared terms for duration and word_count to capture non-linear
    engagement drop-off effects.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    if "duration" in df.columns:
        df["duration_squared"] = df["duration"] ** 2
    if "word_count" in df.columns:
        df["word_count_squared"] = df["word_count"] ** 2
    return df


# ---------------------------------------------------------------------------
# Combined pipeline
# ---------------------------------------------------------------------------

def engineer_features(
    X: pd.DataFrame,
    y: np.ndarray = None,
    encoding_info: dict = None,
    fit: bool = True,
) -> tuple:
    """
    Full feature engineering pipeline.

    When fit=True (training set):
        1. Compute topic target encoding from y.
        2. Apply topic encoding.
        3. Add topic interaction features.
        4. Add linguistic diversity feature.
        5. Add cognitive load feature.
        6. Add polynomial features.
        Returns (X_engineered, encoding_info).

    When fit=False (test set):
        Apply pre-computed encoding_info and add the same derived features.
        Returns (X_engineered, None).

    Parameters
    ----------
    X : pd.DataFrame
    y : np.ndarray or None
        Required when fit=True.
    encoding_info : dict or None
        Required when fit=False.
    fit : bool

    Returns
    -------
    (pd.DataFrame, dict or None)
    """
    if fit:
        assert y is not None, "y must be provided when fit=True"
        if "most_covered_topic" in X.columns:
            encoding_info = compute_topic_encoding(X, y)
            X = apply_topic_encoding(X, encoding_info)
    else:
        if encoding_info is not None and "most_covered_topic" in X.columns:
            X = apply_topic_encoding(X, encoding_info)
        elif "most_covered_topic" in X.columns:
            X = X.drop("most_covered_topic", axis=1)

    X = add_topic_interactions(X)
    X = add_linguistic_features(X)
    X = add_cognitive_load(X)
    X = add_polynomial_features(X)

    n_new = X.shape[1]
    print(f"Feature engineering ({'fit' if fit else 'transform'}) — total features: {n_new}")

    return X, encoding_info
