"""
evaluation.py
-------------
Metric implementations, cross-validation helpers, and visualisation
utilities for the lecture engagement prediction task.

Metrics are implemented from scratch (no sklearn.metrics imports for the
core MCC / FMI calculations) to satisfy the original coursework constraints.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score


# ---------------------------------------------------------------------------
# Label conversion
# ---------------------------------------------------------------------------

def cont_to_class(y_cont: float, threshold: float = 0.2) -> int:
    """
    Convert a continuous engagement score to a binary label.

    Videos with median_engagement >= threshold are labelled as high-engagement
    (class 1); all others as low-engagement (class 0).

    Parameters
    ----------
    y_cont : float
    threshold : float
        Default 0.2 (20% engagement).

    Returns
    -------
    int  (0 or 1)
    """
    return 1 if y_cont >= threshold else 0


def binarise_labels(y_cont: np.ndarray, threshold: float = 0.2) -> np.ndarray:
    """Vectorised version of cont_to_class applied to an array."""
    return np.vectorize(cont_to_class)(y_cont, threshold)


# ---------------------------------------------------------------------------
# Core metrics (implemented from scratch)
# ---------------------------------------------------------------------------

def mcc(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Matthews Correlation Coefficient (MCC).

    Provides a balanced evaluation metric that considers all four quadrants
    of the confusion matrix.  Particularly robust for imbalanced datasets.

    Formula:
        MCC = (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN))

    Range: [-1, 1]
        +1 = perfect prediction
         0 = random prediction
        -1 = complete disagreement

    Parameters
    ----------
    y_actual : np.ndarray
    y_predicted : np.ndarray

    Returns
    -------
    float
    """
    TP = np.sum((y_actual == 1) & (y_predicted == 1))
    TN = np.sum((y_actual == 0) & (y_predicted == 0))
    FP = np.sum((y_actual == 0) & (y_predicted == 1))
    FN = np.sum((y_actual == 1) & (y_predicted == 0))

    numerator = (TP * TN) - (FP * FN)
    denominator = np.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))

    return float(numerator / denominator) if denominator != 0 else 0.0


def fmi(y_actual: np.ndarray, y_predicted: np.ndarray) -> float:
    """
    Fowlkes-Mallows Index (FMI).

    Geometric mean of precision and recall; assesses the model's ability
    to identify high-engagement videos (Class 1).  Penalises extreme
    imbalances between precision and recall.

    Formula:
        FMI = sqrt(Precision * Recall)
            = TP / sqrt((TP+FP)(TP+FN))

    Range: [0, 1]

    Parameters
    ----------
    y_actual : np.ndarray
    y_predicted : np.ndarray

    Returns
    -------
    float
    """
    TP = np.sum((y_actual == 1) & (y_predicted == 1))
    FP = np.sum((y_actual == 0) & (y_predicted == 1))
    FN = np.sum((y_actual == 1) & (y_predicted == 0))

    denom = np.sqrt((TP + FP) * (TP + FN))
    return float(TP / denom) if denom > 0 else 0.0


def regression_metrics(y_actual: np.ndarray, y_predicted: np.ndarray) -> dict:
    """
    Compute MAE, MSE, and R² for a regression model.

    Parameters
    ----------
    y_actual : np.ndarray
    y_predicted : np.ndarray

    Returns
    -------
    dict with keys: mae, mse, r2
    """
    residuals = y_actual - y_predicted
    mae = float(np.mean(np.abs(residuals)))
    mse = float(np.mean(residuals ** 2))
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_actual - np.mean(y_actual)) ** 2)
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return {"mae": mae, "mse": mse, "r2": r2}


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def cross_validate_classifier(model, X, y, cv: int = 5, scoring: str = "matthews_corrcoef"):
    """
    Run k-fold cross-validation and return mean ± std of the scoring metric.

    Parameters
    ----------
    model : sklearn estimator (unfitted)
    X : array-like
    y : array-like
    cv : int
    scoring : str

    Returns
    -------
    (mean_score, std_score)
    """
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    print(f"CV ({cv}-fold) {scoring}: {scores.mean():.4f} ± {scores.std():.4f}")
    return float(scores.mean()), float(scores.std())


# ---------------------------------------------------------------------------
# Evaluation reports
# ---------------------------------------------------------------------------

def evaluate_classifier(
    model,
    X_train,
    y_train,
    X_test,
    y_test,
    model_name: str = "Model",
) -> dict:
    """
    Evaluate a fitted classifier on both training and test sets and print
    a comprehensive report including overfitting analysis.

    Parameters
    ----------
    model : fitted sklearn estimator
    X_train, y_train : training data
    X_test, y_test : test data
    model_name : str

    Returns
    -------
    dict with train / test metrics and performance gaps.
    """
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    train_mcc = mcc(y_train, y_train_pred)
    test_mcc = mcc(y_test, y_test_pred)
    train_fm = fmi(y_train, y_train_pred)
    test_fm = fmi(y_test, y_test_pred)
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    mcc_gap = abs(train_mcc - test_mcc)
    fm_gap = abs(train_fm - test_fm)
    avg_gap = (mcc_gap + fm_gap) / 2

    print(f"\n{'='*50}")
    print(f"  {model_name}")
    print(f"{'='*50}")
    print(f"  {'Metric':<20} {'Train':>8} {'Test':>8} {'Gap':>8}")
    print(f"  {'-'*44}")
    print(f"  {'MCC':<20} {train_mcc:>8.4f} {test_mcc:>8.4f} {mcc_gap:>8.4f}")
    print(f"  {'FM Index':<20} {train_fm:>8.4f} {test_fm:>8.4f} {fm_gap:>8.4f}")
    print(f"  {'Accuracy':<20} {train_acc:>8.4f} {test_acc:>8.4f} {abs(train_acc-test_acc):>8.4f}")

    if avg_gap < 0.05:
        verdict = "No overfitting detected — excellent generalisation."
    elif avg_gap < 0.10:
        verdict = "Minimal overfitting — good generalisation."
    else:
        verdict = "Significant overfitting — consider regularisation or pruning."
    print(f"\n  Overfitting verdict: {verdict}")

    return {
        "model_name": model_name,
        "train_mcc": train_mcc, "test_mcc": test_mcc,
        "train_fm": train_fm, "test_fm": test_fm,
        "train_acc": train_acc, "test_acc": test_acc,
        "mcc_gap": mcc_gap, "fm_gap": fm_gap,
    }


def compare_models(results: list[dict]) -> pd.DataFrame:
    """
    Build a comparison DataFrame from a list of evaluate_classifier outputs.

    Parameters
    ----------
    results : list of dicts (each from evaluate_classifier)

    Returns
    -------
    pd.DataFrame sorted by test_mcc descending.
    """
    df = pd.DataFrame(results)[
        ["model_name", "test_mcc", "test_fm", "test_acc", "mcc_gap", "fm_gap"]
    ]
    df.columns = ["Model", "Test MCC", "Test FM Index", "Test Acc", "MCC Gap", "FM Gap"]
    return df.sort_values("Test MCC", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Visualisation utilities
# ---------------------------------------------------------------------------

def plot_confusion_matrices(
    y_train, y_train_pred, y_test, y_test_pred,
    save_path: str = None,
):
    """
    Side-by-side confusion matrices for train and test sets.

    Parameters
    ----------
    save_path : str or None
        If provided, the figure is saved to this path.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, y_true, y_pred, title, cmap in zip(
        axes,
        [y_train, y_test],
        [y_train_pred, y_test_pred],
        ["Training Set", "Test Set"],
        ["Blues", "Oranges"],
    ):
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap=cmap, ax=ax,
            xticklabels=["Class 0", "Class 1"],
            yticklabels=["Class 0", "Class 1"],
        )
        ax.set_title(f"{title} Confusion Matrix", fontsize=13, fontweight="bold")
        ax.set_ylabel("Actual", fontsize=11)
        ax.set_xlabel("Predicted", fontsize=11)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_model_comparison(results_df: pd.DataFrame, save_path: str = None):
    """
    Bar chart comparing Test MCC and Test FM Index across models.

    Parameters
    ----------
    results_df : pd.DataFrame  (from compare_models)
    save_path : str or None
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = ["#3498DB", "#E74C3C", "#2ECC71", "#9B59B6", "#F39C12"]

    for ax, metric in zip(axes, ["Test MCC", "Test FM Index"]):
        bars = ax.bar(
            range(len(results_df)), results_df[metric],
            color=colors[: len(results_df)], alpha=0.85,
        )
        ax.set_xticks(range(len(results_df)))
        ax.set_xticklabels(results_df["Model"], rotation=30, ha="right", fontsize=10)
        ax.set_ylabel(metric, fontsize=12, fontweight="bold")
        ax.set_title(f"{metric} by Model", fontsize=13, fontweight="bold")
        ax.set_ylim(0, 1.0)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h, f"{h:.3f}",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_feature_importance(
    model,
    feature_names,
    top_n: int = 20,
    engineered_features: list = None,
    save_path: str = None,
):
    """
    Horizontal bar chart of the top-N most important features.
    Engineered features are highlighted in red.

    Parameters
    ----------
    model : fitted tree-based model with feature_importances_ attribute
    feature_names : list of str
    top_n : int
    engineered_features : list of str or None
    save_path : str or None
    """
    importances = pd.DataFrame({
        "feature": feature_names,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False).head(top_n)

    engineered_features = engineered_features or []
    colors = [
        "#E74C3C" if f in engineered_features else "#3498DB"
        for f in importances["feature"]
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(range(len(importances)), importances["importance"], color=colors, alpha=0.85)
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels(importances["feature"], fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("Feature Importance", fontsize=12, fontweight="bold")
    ax.set_title(
        f"Top {top_n} Features  (red = engineered, blue = original)",
        fontsize=13, fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.3, linestyle="--")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


def plot_performance_gap(results: dict, save_path: str = None):
    """
    Bar chart of train/test performance gap per metric.
    Bars are coloured green/orange/red based on gap severity.

    Parameters
    ----------
    results : dict  (from evaluate_classifier)
    save_path : str or None
    """
    metrics = ["MCC", "FM Index"]
    gaps = [results["mcc_gap"], results["fm_gap"]]
    colors = ["#27AE60" if g < 0.05 else "#E67E22" if g < 0.10 else "#E74C3C"
              for g in gaps]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(metrics, gaps, color=colors, alpha=0.8, edgecolor="black")
    ax.axhline(0.05, color="green", linestyle="--", linewidth=1.5, label="Good (<0.05)")
    ax.axhline(0.10, color="orange", linestyle="--", linewidth=1.5, label="Acceptable (<0.10)")
    ax.set_ylabel("|Train - Test|", fontsize=12)
    ax.set_title("Overfitting Analysis: Performance Gap", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, max(gaps) * 1.4 + 0.01)
    for bar, gap in zip(bars, gaps):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{gap:.4f}", ha="center", va="bottom", fontsize=11, fontweight="bold")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
