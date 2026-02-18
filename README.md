# Lecture Engagement Prediction

Predicting **median engagement** (a continuous score between 0 and 1) for online video lectures using a structured, reproducible machine learning workflow.

This project is a compact end-to-end case study covering:
- exploratory data analysis (EDA)
- data preparation and preprocessing
- feature engineering
- training and evaluating regression models
- iterating toward improved solutions

## Problem
Given lecture-level features extracted from metadata / content signals (including transcript-derived characteristics), the goal is to predict:

- **target:** `median_engagement` ∈ [0, 1]  
  where values close to 0 indicate low engagement and values close to 1 indicate high engagement.

## Dataset
The original dataset contains **11,548 observations**, **21 candidate features**, and **1 label** (`median_engagement`).

> Note: this repository is designed to be reproducible without bundling large or restricted datasets.  
> If you have access to the original dataset, place it locally under `data/` (see below).

## Repository structure
```text
lecture-engagement-prediction/
├── notebooks/
│   └── analysis.ipynb
├── src/
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── models.py
│   ├── train.py
│   └── evaluation.py
├── data/
│   ├── README.md
│   └── sample_dataset.csv        # optional small sample for demo
├── results/
│   └── figures/
├── requirements.txt
└── README.md
