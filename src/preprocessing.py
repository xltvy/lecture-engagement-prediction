"""
preprocessing.py
----------------
Data loading, cleaning, missing-value imputation, encoding, scaling,
and train/test splitting for the lecture engagement dataset.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_dataset(data_path: str) -> pd.DataFrame:
    """Load the raw lectures CSV from *data_path* and return a DataFrame."""
    df = pd.read_csv(data_path)
    print(f"Loaded dataset: {df.shape[0]:,} rows x {df.shape[1]} columns")
    return df


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------

def remove_data_quality_errors(df: pd.DataFrame, speed_threshold: float = 5.0) -> pd.DataFrame:
    """
    Remove rows with physically impossible speaker speeds.

    EDA revealed that 2.84% of videos have speaker_speed > 5 words/second,
    which is beyond human speech capability.  Root-cause analysis showed these
    errors originate in erroneous duration / word_count values rather than
    legitimate outliers, so the rows are dropped rather than capped.

    Parameters
    ----------
    df : pd.DataFrame
        Raw lecture dataset.
    speed_threshold : float
        Maximum plausible speaking speed in words/second (default 5.0).

    Returns
    -------
    pd.DataFrame
        Cleaned copy of df.
    """
    before = len(df)
    df = df[df["speaker_speed"] <= speed_threshold].copy()
    removed = before - len(df)
    print(f"Removed {removed} rows with speaker_speed > {speed_threshold} ({removed/before*100:.2f}%)")
    return df


# ---------------------------------------------------------------------------
# Missing-value imputation
# ---------------------------------------------------------------------------

def impute_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values in subject_domain and has_parts.

    Strategy (justified by MAR analysis in EDA):
    - subject_domain : encode missingness as its own category 'unknown',
      preserving the missingness signal.
    - has_parts      : impute False — 95.5% of non-missing values are
      False, so mode imputation introduces minimal bias.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    pd.DataFrame
    """
    df = df.copy()
    df["subject_domain"] = df["subject_domain"].fillna("unknown")
    df["has_parts"] = df["has_parts"].fillna(False)
    return df


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_categorical_features(
    df: pd.DataFrame,
    training_columns: list = None,
) -> tuple:
    """
    Encode categorical features for modelling.

    - most_covered_topic : dropped (2,096 unique Wikipedia URLs —
      one-hot encoding would create an impractically sparse feature space
      and cannot generalise to unseen topics).
    - has_parts          : cast to int (binary 0 / 1).
    - lecture_type       : one-hot encoded (drop_first=True).
    - subject_domain     : one-hot encoded (drop_first=True).

    Parameters
    ----------
    df : pd.DataFrame
    training_columns : list or None
        Column list produced when encoding the training set.  Must be
        supplied when encoding a test / inference set so that the output
        columns are aligned with training (missing dummies are filled with 0,
        extra dummies are dropped).  Pass None when fitting on training data.

    Returns
    -------
    (pd.DataFrame, list)
        Encoded DataFrame and the column list (use as training_columns for
        subsequent test-set calls).
    """
    df = df.copy()

    # Drop high-cardinality URL feature
    if "most_covered_topic" in df.columns:
        df = df.drop("most_covered_topic", axis=1)

    # Binary flag
    df["has_parts"] = df["has_parts"].astype(int)

    # Nominal categoricals -> dummy variables
    df = pd.get_dummies(df, columns=["lecture_type", "subject_domain"], drop_first=True)

    # Align columns with training set (handles unseen / missing rare categories)
    if training_columns is not None:
        df = df.reindex(columns=training_columns, fill_value=0)

    return df, list(df.columns)


# ---------------------------------------------------------------------------
# Scaling
# ---------------------------------------------------------------------------

def scale_numerical_features(
    df: pd.DataFrame,
    scaler=None,
    fit: bool = True,
):
    """
    Apply StandardScaler to all continuous / discrete numeric columns,
    excluding one-hot dummy columns and the target median_engagement.

    Parameters
    ----------
    df : pd.DataFrame
    scaler : StandardScaler or None
        Pass a pre-fitted scaler when fit=False (e.g. for the test set).
    fit : bool
        If True, fit and transform; otherwise only transform.

    Returns
    -------
    (pd.DataFrame, StandardScaler)
        Scaled dataframe and the (possibly newly fitted) scaler.
    """
    df = df.copy()

    # Identify dummy columns (named with a prefix)
    dummy_prefixes = ("lecture_type_", "subject_domain_")
    dummy_cols = [c for c in df.columns if c.startswith(dummy_prefixes)]
    exclude = set(dummy_cols) | {"median_engagement"}

    numerical_cols = [
        c for c in df.columns
        if c not in exclude
        and pd.api.types.is_numeric_dtype(df[c])
        and not pd.api.types.is_bool_dtype(df[c])
    ]

    if fit:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    else:
        df[numerical_cols] = scaler.transform(df[numerical_cols])

    return df, scaler


# ---------------------------------------------------------------------------
# Full preprocessing pipeline
# ---------------------------------------------------------------------------

def preprocess_lecture_dataset(
    dataset: pd.DataFrame,
    scaler=None,
    training_columns: list = None,
    fit: bool = True,
):
    """
    End-to-end preprocessing pipeline.

    Executes the following steps in order:
    1. Remove data-quality errors (impossible speaker speed).
    2. Impute missing values.
    3. Encode categorical features (aligned to training_columns if provided).
    4. Scale numeric features with StandardScaler.

    Parameters
    ----------
    dataset : pd.DataFrame
        Raw lecture dataset (as returned by load_dataset).
    scaler : StandardScaler or None
        Pre-fitted scaler to use when fit=False.
    training_columns : list or None
        Column list from the training run; required when fit=False so that
        one-hot columns are aligned correctly for test / inference data.
    fit : bool
        Whether to fit the scaler on this data (True for train, False for test).

    Returns
    -------
    (pd.DataFrame, StandardScaler, list)
        Preprocessed DataFrame, the fitted scaler, and the column list.
        Store the column list from the training call and pass it as
        training_columns on subsequent test-set calls.
    """
    df = remove_data_quality_errors(dataset)
    df = impute_missing_values(df)
    df, columns = encode_categorical_features(df, training_columns=training_columns)
    df, scaler = scale_numerical_features(df, scaler=scaler, fit=fit)
    print(f"Preprocessing complete — final shape: {df.shape}")
    return df, scaler, columns


# ---------------------------------------------------------------------------
# Feature / label extraction and splitting
# ---------------------------------------------------------------------------

def prepare_features_labels(
    preprocessed_df: pd.DataFrame,
    target_col: str = "median_engagement",
):
    """
    Split a preprocessed DataFrame into feature matrix X and label vector y.

    Parameters
    ----------
    preprocessed_df : pd.DataFrame
    target_col : str

    Returns
    -------
    (pd.DataFrame, np.ndarray)
    """
    X = preprocessed_df.drop(target_col, axis=1)
    y = preprocessed_df[target_col].values
    return X, y


def split_data(
    X: pd.DataFrame,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify=None,
):
    """
    Wrapper around sklearn's train_test_split.

    Parameters
    ----------
    X : pd.DataFrame
    y : np.ndarray
    test_size : float
    random_state : int
    stratify : array-like or None
        Pass y (or a binary version of it) to preserve class proportions.

    Returns
    -------
    (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    print(
        f"Split — train: {len(X_train):,} samples, test: {len(X_test):,} samples "
        f"({test_size*100:.0f}% held out)"
    )
    return X_train, X_test, y_train, y_test
