"""Build submission.csv for the (closed) ISIC-2024 competition.

The competition is over; late submissions still score against the hidden private
LB. This CLI loads the frozen per-fold GBDT boosters, predicts on the held-out
test metadata, optionally rank-blends image-expert test scores, and writes the
two-column submission (isic_id, target).

Inference mirrors the leak-free training transform exactly: each booster carries
its own per-fold feature `state`, and `predict_gbdt` reapplies that state before
predicting, then averages across folds.

Usage:
    python -m src.submit                          # GBDT-only, default paths
    python -m src.submit --cfg configs/default.yaml
    python -m src.submit --boosters experiments/gbdt_boosters.joblib \\
                         --out submission.csv
    python -m src.submit --image-scores experiments/vision_test_scores.parquet \\
                         --image-weight 0.5       # rank-blend GBDT + image
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import features as F
from .data import ID


def predict_gbdt(test_meta: pd.DataFrame, boosters: list) -> np.ndarray:
    """Average predictions across the per-fold boosters (booster, cols, state)."""
    preds = []
    for booster, cols, state in boosters:
        t = F.transform_fold_features(test_meta, state)
        preds.append(booster.predict(t[cols], num_iteration=booster.best_iteration))
    return np.mean(preds, axis=0)


def load_image_scores(path: str | Path, test_meta: pd.DataFrame) -> np.ndarray:
    """Load image-expert test scores, aligned to `test_meta` row order.

    Expects a parquet with columns [isic_id, <score>] where <score> is the only
    non-id column (e.g. produced by averaging the per-fold image checkpoints over
    the test set). Raises if the contract is unmet or rows are missing.

    NOTE: producing this file requires the trained per-fold image checkpoints,
    which are generated later on GPU (see src/vision/train.py). Until those exist,
    this hook is intentionally unwired — the path below documents how to fill it,
    rather than fabricating image predictions. Run GBDT-only in the meantime.
    """
    df = pd.read_parquet(path)
    if ID not in df.columns:
        raise ValueError(f"{path}: missing {ID!r} column")
    score_cols = [c for c in df.columns if c != ID]
    if len(score_cols) != 1:
        raise ValueError(
            f"{path}: expected exactly one score column besides {ID!r}, "
            f"got {score_cols}"
        )
    merged = test_meta[[ID]].merge(df[[ID, score_cols[0]]], on=ID, how="left")
    if merged[score_cols[0]].isna().any():
        n = int(merged[score_cols[0]].isna().sum())
        raise ValueError(f"{path}: {n} test rows have no image score")
    return merged[score_cols[0]].to_numpy(dtype=float)


def rank_blend(scores: list[np.ndarray], weights: list[float] | None = None) -> np.ndarray:
    """Weighted average of rank-normalized scores (scale-free, robust)."""
    if weights is None:
        weights = [1.0] * len(scores)
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    ranked = np.vstack([rankdata(s) / len(s) for s in scores])
    return np.average(ranked, axis=0, weights=w)


def predict_test(test_meta: pd.DataFrame, boosters: list,
                 image_scores_path: str | Path | None = None,
                 image_weight: float = 0.5) -> np.ndarray:
    """GBDT test scores, optionally rank-blended with image-expert scores."""
    gbdt = predict_gbdt(test_meta, boosters)
    if not image_scores_path:
        return gbdt
    img = load_image_scores(image_scores_path, test_meta)
    gbdt_w = max(0.0, 1.0 - float(image_weight))
    return rank_blend([gbdt, img], weights=[gbdt_w, float(image_weight)])


def write_submission(test_meta: pd.DataFrame, scores: np.ndarray,
                     path: str | Path = "submission.csv") -> None:
    assert len(test_meta) == len(scores)
    out = pd.DataFrame({ID: test_meta[ID].values, "target": scores})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(f"wrote {path}  ({len(out)} rows)")


def main(argv: list[str] | None = None) -> None:
    import joblib

    from . import config as C

    ap = argparse.ArgumentParser(description="Build submission.csv from frozen models.")
    ap.add_argument("--cfg", default="configs/default.yaml",
                    help="config providing paths.test_meta")
    ap.add_argument("--test-meta", default=None,
                    help="override test metadata CSV (else cfg paths.test_meta)")
    ap.add_argument("--boosters", default="experiments/gbdt_boosters.joblib",
                    help="joblib list of (booster, cols, state) tuples")
    ap.add_argument("--image-scores", default=None,
                    help="optional parquet [isic_id, score] from image experts; "
                         "if omitted, GBDT-only")
    ap.add_argument("--image-weight", type=float, default=0.5,
                    help="rank-blend weight on the image scores (0..1)")
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args(argv)

    cfg = C.load(a.cfg)
    test_meta_path = a.test_meta or cfg.paths.test_meta
    test_meta = pd.read_csv(test_meta_path, low_memory=False)
    assert ID in test_meta.columns, f"{test_meta_path}: missing {ID!r}"

    boosters = joblib.load(a.boosters)
    scores = predict_test(test_meta, boosters,
                          image_scores_path=a.image_scores,
                          image_weight=a.image_weight)
    write_submission(test_meta, scores, a.out)
    mode = ("GBDT + image rank-blend" if a.image_scores else "GBDT-only")
    print(f"submission mode: {mode}  ({len(boosters)} folds)")


if __name__ == "__main__":
    main()
