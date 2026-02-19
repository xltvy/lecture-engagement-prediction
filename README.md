# Lecture Engagement Prediction

Predicting **median engagement** (a continuous score 0–1) for online video lectures using a clean, modular, and reproducible machine learning pipeline.

## Problem

Given features extracted from video lecture transcripts and metadata, predict `median_engagement` — the median fraction of the video watched by viewers.

Two tasks are addressed:
- **Regression** — predict the continuous engagement score directly (Ridge Regression).
- **Binary classification** — predict whether engagement ≥ 0.20 (Logistic Regression, Random Forest, XGBoost).

## Dataset

11,548 video lectures · 21 features · 1 target label (11,220 after removing 328 data-quality errors).

Features include linguistic rates (auxiliary verbs, pronouns, entropy), content metadata (duration, word count, freshness), speaker characteristics (speed, silence rate), and categorical labels (lecture type, subject domain).

> Place the raw CSV at `data/lectures_dataset.csv` before running any code.

## Repository Structure

```
lecture-engagement-prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── analysis.ipynb          # Clean analysis report — EDA → modelling → evaluation
│
├── src/
│   ├── preprocessing.py        # Loading, cleaning, imputation, encoding, scaling
│   ├── feature_engineering.py  # Target encoding, interaction and polynomial features
│   ├── models.py               # Ridge, Logistic, Random Forest, XGBoost, Ensemble
│   ├── evaluation.py           # MCC, FMI, confusion matrices, comparison plots
│   └── train.py                # CLI entry point with argparse + reproducibility seed
│
├── data/
│   └── lectures_dataset.csv    # (not tracked by git — add your own copy)
│
└── results/
    └── figures/                # Plots saved by the notebook / train.py
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full training pipeline from the command line
python src/train.py --data data/lectures_dataset.csv --model rf --seed 42

# 3. Or open the analysis notebook
jupyter notebook notebooks/analysis.ipynb
```

### CLI Options (`src/train.py`)

| Flag | Default | Description |
|---|---|---|
| `--data` | `data/lectures_dataset.csv` | Path to raw CSV |
| `--model` | `logistic` | `ridge` · `logistic` · `rf` · `xgb` · `ensemble` |
| `--tune` | off | Run RandomizedSearchCV hyperparameter tuning (rf / xgb only) |
| `--n-iter` | `20` | Number of configs sampled during tuning |
| `--seed` | `42` | Global random seed |
| `--threshold` | `0.2` | Binarisation threshold for classification |
| `--test-size` | `0.2` | Fraction held out for testing |
| `--no-feature-engineering` | off | Skip feature engineering, use preprocessed features only |
| `--output` | `results/` | Directory to save evaluation artefacts |
| `--save-figures` | off | Save evaluation plots to `--output/figures/` |

## Module Overview

| Module | Responsibility |
|---|---|
| `preprocessing.py` | Data quality filtering, missing-value imputation, categorical encoding, StandardScaler |
| `feature_engineering.py` | Bayesian target encoding of topics, interaction features, cognitive load, polynomial terms |
| `models.py` | Model constructors and default hyperparameter configs for all supported models |
| `evaluation.py` | MCC and FMI (from scratch), regression metrics, overfitting analysis, visualisation helpers |
| `train.py` | End-to-end CLI pipeline with reproducibility seeding |

## Key Results

| Model | Test MCC | Test FM Index | MCC Gap | Overfitting |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.514 | 0.612 | 0.022 | None |
| **Random Forest** | **0.610** | **0.675** | 0.191 | Significant |
| XGBoost | 0.572 | 0.638 | 0.412 | Significant |
| Voting Ensemble (RF + XGB) | 0.576 | 0.644 | 0.369 | Significant |

Random Forest achieves the best test performance (+18.7% MCC over the logistic baseline). All tree-based models show significant overfitting; the ensemble does not outperform Random Forest in isolation. Topic-derived engineered features (`topic_duration_interaction`, `topic_content_interaction`, `topic_engagement_score`) occupy 3 of the top 5 feature importance positions.

## Reproducibility

All scripts accept `--seed` (default 42). Seeds are set for Python `random`, NumPy, and optionally PyTorch / TensorFlow at startup.
