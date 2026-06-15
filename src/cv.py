"""
cv.py — the locked validation spine for the ISIC-2024 SLICE-3D project.

Single source of truth for two things, and NOTHING in the repo is allowed to
recompute either of them independently:

  1. the official ISIC-2024 primary metric (partial AUC above 80% TPR), and
  2. patient-grouped, target-stratified fold assignments.

Every model — tabular expert, image expert, combiner — reads the SAME frozen
fold file produced by `freeze()`. That is what makes the efficiency Pareto
comparison valid: identical splits for every point on the curve.

Run once to build the folds:
    python -m src.cv --meta data/train-metadata.csv --out data/folds.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import auc, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

# --- frozen config --------------------------------------------------------
SEED = 42
N_SPLITS = 5
ID = "isic_id"
GROUP = "patient_id"
TARGET = "target"

# The Kaggle competition ran at min_tpr = 0.80  -> scores in [0, 0.20].
# The official metrics-repo README states 0.88  -> scores in [0, 0.12].
# Use 0.80 for any leaderboard-comparable number; confirm against the repo's
# PrimaryMetric-pAUC.py before locking final paper figures.
MIN_TPR = 0.80


def pauc_above_tpr(y_true: np.ndarray, y_score: np.ndarray, min_tpr: float = MIN_TPR) -> float:
    """Official ISIC-2024 primary metric: partial AUC above `min_tpr` TPR.

    Mirrors the competition scoring notebook. Labels and scores are flipped so
    the TPR >= min_tpr region maps to FPR <= (1 - min_tpr); the partial area is
    then integrated over that FPR band. Returns a value in [0, 1 - min_tpr]
    (i.e. [0, 0.20] at min_tpr = 0.80). Higher is better.
    """
    y_true = np.asarray(y_true, dtype=float).ravel()
    y_score = np.asarray(y_score, dtype=float).ravel()

    v_gt = np.abs(y_true - 1)        # flip labels: 1 -> 0, 0 -> 1
    v_pred = -1.0 * y_score          # flip scores to their complement
    max_fpr = abs(1.0 - min_tpr)

    fpr, tpr, _ = roc_curve(v_gt, v_pred)
    if max_fpr >= 1.0:
        return float(auc(fpr, tpr))

    stop = np.searchsorted(fpr, max_fpr, side="right")
    x_interp = [fpr[stop - 1], fpr[stop]]
    y_interp = [tpr[stop - 1], tpr[stop]]
    tpr = np.append(tpr[:stop], np.interp(max_fpr, x_interp, y_interp))
    fpr = np.append(fpr[:stop], max_fpr)
    return float(auc(fpr, tpr))


def make_folds(meta: pd.DataFrame, n_splits: int = N_SPLITS, seed: int = SEED) -> pd.Series:
    """Patient-grouped, target-stratified fold indices aligned to meta.index.

    No patient ever straddles two folds. Raises if that guarantee is violated.
    """
    assert GROUP in meta.columns and TARGET in meta.columns, (
        f"metadata must contain {GROUP!r} and {TARGET!r}"
    )
    folds = pd.Series(-1, index=meta.index, dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for k, (_, val_idx) in enumerate(sgkf.split(meta, meta[TARGET], groups=meta[GROUP])):
        folds.iloc[val_idx] = k
    assert (folds >= 0).all(), "some rows were never assigned to a fold"
    _assert_no_leak(meta, folds)
    return folds


def _assert_no_leak(meta: pd.DataFrame, folds: pd.Series) -> None:
    """Hard guarantee: every patient lives in exactly one fold."""
    n_folds_per_patient = (
        meta.assign(_f=folds.values).groupby(GROUP)["_f"].nunique()
    )
    leaked = n_folds_per_patient[n_folds_per_patient > 1]
    if len(leaked):
        raise RuntimeError(
            f"PATIENT LEAK: {len(leaked)} patient(s) span >1 fold. "
            "Refusing to proceed — the CV contract is violated."
        )


def oof_pauc(y_true: np.ndarray, oof_pred: np.ndarray, min_tpr: float = MIN_TPR) -> float:
    """Score full out-of-fold predictions with the official metric."""
    return pauc_above_tpr(y_true, oof_pred, min_tpr)


def load_folds(path: str | Path = "data/folds.parquet") -> pd.DataFrame:
    """Read the frozen fold assignments. Every model uses this — do not rebuild."""
    return pd.read_parquet(path)


def freeze(meta_path: str | Path, out_path: str | Path = "data/folds.parquet") -> pd.DataFrame:
    """Build folds once and persist them. Single source of truth on disk."""
    meta = pd.read_csv(meta_path, low_memory=False)
    folds = make_folds(meta)
    out = meta[[ID, GROUP, TARGET]].copy()
    out["fold"] = folds.values
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return out


def _report(out: pd.DataFrame) -> None:
    n_pat = out[GROUP].nunique()
    n_pos = int(out[TARGET].sum())
    print(
        f"rows={len(out):,}  patients={n_pat:,}  positives={n_pos:,}  "
        f"prevalence={n_pos / len(out):.4%}"
    )
    g = out.groupby("fold").agg(
        rows=(ID, "size"), patients=(GROUP, "nunique"), pos=(TARGET, "sum")
    )
    g["pos_rate"] = g["pos"] / g["rows"]
    print("\nper-fold:\n" + g.to_string())
    print("\nno-leak check: PASSED (enforced in make_folds)")

    # sanity: a random score = area under the ROC diagonal over FPR in [0, 1-min_tpr]
    # = (1-min_tpr)^2 / 2, i.e. ~0.02 at min_tpr=0.80. Perfect = 0.20. Higher is better.
    rng = np.random.default_rng(SEED)
    rand = pauc_above_tpr(out[TARGET].values, rng.random(len(out)))
    print(f"\nrandom-score pAUC@{MIN_TPR:.2f}TPR sanity = {rand:.4f}  (expect ~0.02; ceiling 0.20)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Freeze ISIC-2024 CV folds + sanity-check the metric.")
    ap.add_argument("--meta", default="data/train-metadata.csv")
    ap.add_argument("--out", default="data/folds.parquet")
    args = ap.parse_args()
    _report(freeze(args.meta, args.out))
