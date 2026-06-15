"""Tests for the validation spine. Run: pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src import cv


def test_metric_anchors():
    rng = np.random.default_rng(0)
    n = 200_000
    y = (rng.random(n) < 0.001).astype(int)
    perfect = y + rng.random(n) * 1e-6
    assert abs(cv.pauc_above_tpr(y, perfect) - 0.20) < 1e-3      # ceiling
    assert abs(cv.pauc_above_tpr(y, rng.random(n)) - 0.02) < 5e-3  # random ~ diagonal
    assert cv.pauc_above_tpr(y, -perfect) < 1e-3                  # inverted ~ 0


def test_folds_no_leak():
    rng = np.random.default_rng(1)
    n, n_pat = 50_000, 400
    pid = rng.integers(0, n_pat, n)
    meta = pd.DataFrame({
        cv.ID: [f"ISIC_{i:07d}" for i in range(n)],
        cv.GROUP: [f"P{p:04d}" for p in pid],
        cv.TARGET: 0,
    })
    meta.loc[rng.choice(n, 100, replace=False), cv.TARGET] = 1
    folds = cv.make_folds(meta)
    # every patient in exactly one fold
    per_patient = meta.assign(f=folds.values).groupby(cv.GROUP)["f"].nunique()
    assert (per_patient == 1).all()


def test_leak_raises():
    meta = pd.DataFrame({
        cv.ID: ["a", "b", "c", "d"],
        cv.GROUP: ["P0", "P0", "P1", "P1"],
        cv.TARGET: [1, 0, 0, 1],
    })
    leaky = pd.Series([0, 1, 0, 1], index=meta.index)  # P0 split across folds
    with pytest.raises(RuntimeError):
        cv._assert_no_leak(meta, leaky)


def test_matches_official_metric():
    """cv.pauc_above_tpr must equal the vendored official score() to float tol."""
    from src import metric_official as M

    rng = np.random.default_rng(7)
    for _ in range(20):
        n = rng.integers(2000, 8000)
        y = (rng.random(n) < rng.uniform(0.001, 0.02)).astype(int)
        if y.sum() == 0:           # need at least one positive for a valid ROC
            y[rng.integers(n)] = 1
        pred = rng.random(n)
        mine = cv.pauc_above_tpr(y, pred, min_tpr=0.80)
        sol = pd.DataFrame({"id": np.arange(n), "target": y})
        sub = pd.DataFrame({"id": np.arange(n), "target": pred})
        official = M.score(sol, sub, "id", min_tpr=0.80)
        assert abs(mine - official) < 1e-9, (mine, official)
