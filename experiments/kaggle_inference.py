"""Late-submission inference for the ISIC-2024 Kaggle code competition.

The private score is only produced by a notebook run on the hidden test set. To use
this: upload this repository and experiments/gbdt_boosters.joblib as a Kaggle dataset,
attach the competition data, and run in a Kaggle notebook. It reapplies the leak-free
per-fold feature transform, averages the bagged boosters, and writes submission.csv.

    python experiments/kaggle_inference.py \
        --test data/test-metadata.csv --boosters experiments/gbdt_boosters.joblib
"""
from __future__ import annotations

import argparse

import pandas as pd

from src import submit


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="/kaggle/input/isic-2024-challenge/test-metadata.csv")
    ap.add_argument("--boosters", default="experiments/gbdt_boosters.joblib")
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args()

    import joblib
    test = pd.read_csv(a.test, low_memory=False)
    boosters = joblib.load(a.boosters)
    scores = submit.predict_gbdt(test, boosters)
    submit.write_submission(test, scores, a.out)


if __name__ == "__main__":
    main()
