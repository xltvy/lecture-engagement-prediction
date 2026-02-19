"""
train.py
--------
Command-line entry point for training lecture engagement prediction models.

Usage examples
--------------
# Train Ridge regression (regression task) with default settings:
    python src/train.py --data data/lectures_dataset.csv --model ridge

# Train logistic regression (binary classification):
    python src/train.py --data data/lectures_dataset.csv --model logistic

# Train Random Forest with hyperparameter tuning:
    python src/train.py --data data/lectures_dataset.csv --model rf --tune

# Train XGBoost with a custom random seed:
    python src/train.py --data data/lectures_dataset.csv --model xgb --seed 99

# Save results to a custom directory:
    python src/train.py --data data/lectures_dataset.csv --model rf \\
        --output results/ --seed 42
"""

import argparse
import sys
import os
import random
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix random seeds for Python, NumPy, and (optionally) PyTorch / TF."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
    except ImportError:
        pass
    print(f"[seed] Global random seed set to {seed}")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Train a lecture engagement prediction model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- Data ---
    parser.add_argument(
        "--data",
        type=str,
        default="data/lectures_dataset.csv",
        help="Path to the raw lectures CSV file.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of data reserved for the test set.",
    )

    # --- Model ---
    parser.add_argument(
        "--model",
        type=str,
        default="logistic",
        choices=["ridge", "logistic", "rf", "xgb", "ensemble"],
        help=(
            "Model type: 'ridge' (regression), 'logistic' (binary classification), "
            "'rf' (Random Forest), 'xgb' (XGBoost), 'ensemble' (RF + XGBoost voting)."
        ),
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        default=False,
        help="Run RandomizedSearchCV hyperparameter tuning (rf / xgb only).",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=20,
        help="Number of parameter settings sampled during tuning.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="Regularisation strength for Ridge regression.",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="Inverse regularisation strength for Logistic regression.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.2,
        help="Engagement threshold for binarising labels (classification tasks).",
    )

    # --- Feature engineering ---
    parser.add_argument(
        "--no-feature-engineering",
        action="store_true",
        default=False,
        help="Skip feature engineering (use preprocessed features as-is).",
    )

    # --- Reproducibility ---
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Global random seed for reproducibility.",
    )

    # --- Output ---
    parser.add_argument(
        "--output",
        type=str,
        default="results/",
        help="Directory to save evaluation artefacts (figures, metrics CSV).",
    )
    parser.add_argument(
        "--save-figures",
        action="store_true",
        default=False,
        help="Save evaluation plots to --output/figures/.",
    )

    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------

def main(argv=None):
    args = parse_args(argv)

    # 1. Reproducibility
    set_seed(args.seed)

    # 2. Import project modules (resolved relative to this file's location)
    src_dir = Path(__file__).parent
    sys.path.insert(0, str(src_dir))

    from preprocessing import load_dataset, preprocess_lecture_dataset, prepare_features_labels, split_data
    from feature_engineering import engineer_features
    from models import (
        train_ridge_model, train_logistic_model,
        train_random_forest, train_xgboost, build_voting_ensemble,
    )
    from evaluation import (
        binarise_labels, evaluate_classifier, regression_metrics,
        plot_confusion_matrices, plot_feature_importance, plot_performance_gap,
    )

    # 3. Load and preprocess
    print(f"\n[1/5] Loading data from '{args.data}'...")
    raw_df = load_dataset(args.data)

    print("\n[2/5] Preprocessing...")
    preprocessed_df, scaler, train_columns = preprocess_lecture_dataset(raw_df, fit=True)

    X_full, y_cont = prepare_features_labels(preprocessed_df)

    # 4. Label preparation and train/test split (BEFORE feature engineering
    #    to prevent any future leakage from data-dependent transforms)
    print("\n[3/5] Splitting data...")
    is_classification = args.model != "ridge"

    if is_classification:
        y = binarise_labels(y_cont, threshold=args.threshold)
        stratify = y
    else:
        y = y_cont
        stratify = None

    X_train_raw, X_test_raw, y_train, y_test = split_data(
        X_full, y, test_size=args.test_size, random_state=args.seed, stratify=stratify
    )

    # 5. Feature engineering — fit on training set only, apply to test set
    if not args.no_feature_engineering:
        print("\n[4/5] Feature engineering...")
        # Note: most_covered_topic is dropped during preprocessing so topic
        # encoding is unavailable in this CLI pipeline; the 4 non-topic features
        # (entropy_per_word, cognitive_load, duration_squared, word_count_squared)
        # are still created. See notebooks/analysis.ipynb for the full
        # leakage-safe pipeline with topic encoding.
        X_train, encoding_info = engineer_features(X_train_raw, y=y_train, fit=True)
        X_test, _              = engineer_features(X_test_raw, encoding_info=encoding_info, fit=False)
    else:
        print("\n[4/5] Skipping feature engineering (--no-feature-engineering).")
        X_train, X_test = X_train_raw, X_test_raw
        encoding_info = None

    # 6. Train
    print(f"\n[5/5] Training model: {args.model}...")

    output_dir = Path(args.output)
    fig_dir = output_dir / "figures"
    if args.save_figures:
        fig_dir.mkdir(parents=True, exist_ok=True)

    if args.model == "ridge":
        model = train_ridge_model(X_train, y_train, hyperparams={"alpha": args.alpha, "random_state": args.seed})
        y_pred = model.predict(X_test)
        metrics = regression_metrics(y_test, y_pred)
        print(f"\nTest metrics — MAE: {metrics['mae']:.4f}, MSE: {metrics['mse']:.4f}, R²: {metrics['r2']:.4f}")

    elif args.model == "logistic":
        model = train_logistic_model(
            X_train, y_train,
            hyperparams={"C": args.C, "random_state": args.seed, "max_iter": 1000,
                         "solver": "lbfgs", "class_weight": "balanced"},
        )
        results = evaluate_classifier(model, X_train, y_train, X_test, y_test, "Logistic Regression")
        if args.save_figures:
            plot_confusion_matrices(
                y_train, model.predict(X_train), y_test, model.predict(X_test),
                save_path=str(fig_dir / "confusion_matrix_logistic.png"),
            )
            plot_performance_gap(results, save_path=str(fig_dir / "gap_logistic.png"))

    elif args.model == "rf":
        model = train_random_forest(
            X_train, y_train, tune=args.tune, n_iter=args.n_iter, random_state=args.seed
        )
        results = evaluate_classifier(model, X_train, y_train, X_test, y_test, "Random Forest")
        if args.save_figures:
            engineered = [
                "topic_engagement_score", "topic_duration_interaction",
                "topic_content_interaction", "entropy_per_word",
                "cognitive_load", "duration_squared", "word_count_squared",
            ]
            plot_feature_importance(
                model, X_train.columns.tolist(), engineered_features=engineered,
                save_path=str(fig_dir / "feature_importance_rf.png"),
            )
            plot_confusion_matrices(
                y_train, model.predict(X_train), y_test, model.predict(X_test),
                save_path=str(fig_dir / "confusion_matrix_rf.png"),
            )

    elif args.model == "xgb":
        pos = np.sum(y_train == 1)
        neg = np.sum(y_train == 0)
        spw = neg / pos if pos > 0 else 1.0
        model = train_xgboost(
            X_train, y_train, scale_pos_weight=spw,
            tune=args.tune, n_iter=args.n_iter, random_state=args.seed,
        )
        results = evaluate_classifier(model, X_train, y_train, X_test, y_test, "XGBoost")
        if args.save_figures:
            plot_confusion_matrices(
                y_train, model.predict(X_train), y_test, model.predict(X_test),
                save_path=str(fig_dir / "confusion_matrix_xgb.png"),
            )

    elif args.model == "ensemble":
        pos = np.sum(y_train == 1)
        neg = np.sum(y_train == 0)
        spw = neg / pos if pos > 0 else 1.0

        rf = train_random_forest(X_train, y_train, tune=args.tune, random_state=args.seed)
        xgb_m = train_xgboost(X_train, y_train, scale_pos_weight=spw, tune=args.tune, random_state=args.seed)

        model = build_voting_ensemble(rf, xgb_m)
        model.fit(X_train, y_train)

        results = evaluate_classifier(model, X_train, y_train, X_test, y_test, "Voting Ensemble")
        if args.save_figures:
            plot_confusion_matrices(
                y_train, model.predict(X_train), y_test, model.predict(X_test),
                save_path=str(fig_dir / "confusion_matrix_ensemble.png"),
            )

    print("\nDone.")
    return model


if __name__ == "__main__":
    main()
