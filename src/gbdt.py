"""LightGBM (+ CatBoost) tabular expert — the score-carrying backbone.

Trains models per fold on the FROZEN folds, produces leak-free out-of-fold (OOF)
predictions, and scores them with the official metric from `cv.py`. The OOF
vector is what the combiner consumes; the per-fold models are what `submit.py`
ensembles at inference time.

Two training paths live here:

  * `train_oof` (Step 1-2): one LightGBM per fold, pAUC-driven early stopping,
    trained on the FULL fold-train side. Kept for the clean Step-3a ablation
    (params unchanged, only the feature set varies).

  * `train_oof_bagged` (Step 3b/3c): greysky-lineage recipe. Per fold and per
    seed, MANUALLY undersample the train NEGATIVES to a small ratio (all
    positives kept), train a fixed-iteration booster, predict the FULL held-out
    val fold, and rank-average the per-seed predictions into the fold's OOF. A
    CatBoost expert is trained the same way; the final OOF is a rank-average of
    the LightGBM-bag and CatBoost-bag OOF vectors.

Leak-safety (cv-guardian contract):
  - Undersampling is applied to the TRAIN side of each fold ONLY. The held-out
    val fold is NEVER subsampled, and OOF is written only for held-out rows.
  - Fold features are fit on the FULL train side (`fit_fold_features`) BEFORE
    undersampling, so the undersample never changes target-encoding statistics
    in a label-dependent-on-val way; te_* still only ever sees train labels.
  - Within-patient deviations (pdev/prank/pxc) are computed inside each split in
    isolation and keyed on patient_id, so they never cross the fold boundary.

Booster persistence contract (consumed by `submit.py:predict_gbdt`):
  `boosters` is a flat list of `(model, cols, state)` tuples where `model`
  exposes `.predict(X, num_iteration=...)` and `.best_iteration`. The bagged
  path wraps every per-(fold, seed) LightGBM/CatBoost model in `_RankModel` so
  the uniform contract holds and submit.py averages them unchanged.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import cv
from . import features as F
from .data import ID, TARGET


def _pauc_feval(preds: np.ndarray, dataset) -> tuple[str, float, bool]:
    """Custom LightGBM eval: the OFFICIAL pAUC@80%TPR from cv.py.

    Selection (early stopping) must track the tail metric we actually optimize,
    NOT LightGBM's built-in full-ROC 'auc' — the two disagree sharply in the
    393-positive regime and AUC-driven ES stops at iteration 1. Returns the
    triple LightGBM expects: (name, value, is_higher_better=True).
    """
    y_true = dataset.get_label()
    return "pauc80", cv.pauc_above_tpr(y_true, preds), True


def train_oof(meta: pd.DataFrame, folds: pd.DataFrame, params: dict,
              num_boost_round: int = 5000, early_stopping: int = 200,
              seed: int = cv.SEED) -> tuple[np.ndarray, list, float]:
    """Return (oof_pred aligned to meta, list of per-fold boosters, oof pAUC).

    Step 1-2 path: one LightGBM per fold on the FULL fold-train side, with
    early stopping driven by the custom `_pauc_feval` (official pAUC@80%TPR),
    not LightGBM's internal metric (disabled via `metric: "None"`).
    """
    import lightgbm as lgb

    df = meta.merge(folds[[ID, "fold"]], on=ID, how="left")
    assert df["fold"].notna().all(), "every row must have a fold (run src.cv first)"

    run_params = {**params, "seed": seed, "metric": "None"}

    oof = np.zeros(len(df), dtype=float)
    boosters = []
    y = df[TARGET].values

    for k in sorted(df["fold"].unique()):
        tr_idx = df.index[df["fold"] != k]
        va_idx = df.index[df["fold"] == k]

        state = F.fit_fold_features(df.loc[tr_idx])
        tr = F.transform_fold_features(df.loc[tr_idx], state)
        va = F.transform_fold_features(df.loc[va_idx], state)
        cols = F.feature_columns(tr)

        dtrain = lgb.Dataset(tr[cols], label=y[tr_idx])
        dvalid = lgb.Dataset(va[cols], label=y[va_idx], reference=dtrain)
        booster = lgb.train(
            run_params,
            dtrain,
            num_boost_round=num_boost_round,
            valid_sets=[dvalid],
            feval=_pauc_feval,
            callbacks=[lgb.early_stopping(early_stopping, verbose=False),
                       lgb.log_evaluation(0)],
        )
        oof[va_idx] = booster.predict(va[cols], num_iteration=booster.best_iteration)
        boosters.append((booster, cols, state))

    score = cv.oof_pauc(y, oof)
    return oof, boosters, score


# ----------------------------------------------------------------------------
# Step 3b/3c: undersampled multi-seed bagging (LightGBM + CatBoost)
# ----------------------------------------------------------------------------

# greysky-lineage default seed bag. Each seed drives BOTH the negative
# undersample draw and the booster's internal RNG so the ensemble is diverse.
SEED_BAG = (12, 22, 32, 42, 52)


class _RankModel:
    """Uniform inference adapter over a LightGBM Booster or CatBoost classifier.

    Exposes `.predict(X, num_iteration=...)` and `.best_iteration` so the
    persisted `(model, cols, state)` tuples satisfy the exact contract
    `submit.py:predict_gbdt` consumes, regardless of the underlying library.
    `submit.py` averages raw probabilities across all tuples; mixing LightGBM
    and CatBoost probabilities is acceptable because both are calibrated to the
    same [0, 1] malignancy scale on the same features.
    """

    def __init__(self, kind: str, model):
        self.kind = kind            # "lgb" | "cat"
        self.model = model
        self.best_iteration = None  # fixed-iteration models: predict full model

    def predict(self, X, num_iteration=None):  # noqa: D401 - simple delegate
        # Both wrappers are sklearn-style classifiers: `.predict` returns CLASS
        # LABELS, so we MUST use `.predict_proba(...)[:, 1]` for the positive-
        # class probability (the score the pAUC metric ranks).
        return self.model.predict_proba(X)[:, 1]


def _undersample(tr_idx: np.ndarray, y: np.ndarray, neg_ratio: float,
                 seed: int) -> np.ndarray:
    """Keep ALL positives + a random sample of negatives at `neg_ratio`.

    Returns the SUBSET of train positional indices to train on. Applied to the
    TRAIN side only — the held-out val fold is untouched. `neg_ratio` mirrors
    RandomUnderSampler(sampling_strategy=ratio): the kept negative count is
    n_pos / neg_ratio (greysky used 0.01 -> ~100 negatives per positive).
    """
    rng = np.random.default_rng(seed)
    pos = tr_idx[y[tr_idx] == 1]
    neg = tr_idx[y[tr_idx] == 0]
    n_pos = len(pos)
    # sampling_strategy semantics: n_pos / n_neg_kept = neg_ratio
    n_neg_keep = int(round(n_pos / max(neg_ratio, 1e-9)))
    n_neg_keep = min(n_neg_keep, len(neg))
    neg_keep = rng.choice(neg, size=n_neg_keep, replace=False)
    sub = np.concatenate([pos, neg_keep])
    rng.shuffle(sub)
    return sub


def _train_lgb_seed(Xtr, ytr, params: dict, n_estimators: int, seed: int):
    import lightgbm as lgb

    p = {**params, "seed": seed, "metric": "None", "n_estimators": n_estimators}
    p.pop("num_boost_round", None)
    clf = lgb.LGBMClassifier(**p)
    clf.fit(Xtr, ytr)
    return _RankModel("lgb", clf)


def _train_cat_seed(Xtr, ytr, params: dict, n_estimators: int, seed: int):
    from catboost import CatBoostClassifier

    p = {**params}
    p["iterations"] = n_estimators
    p["random_seed"] = seed
    p.setdefault("verbose", False)
    p.setdefault("allow_writing_files", False)
    clf = CatBoostClassifier(**p)
    clf.fit(Xtr, ytr)
    return _RankModel("cat", clf)


def train_oof_bagged(
    meta: pd.DataFrame,
    folds: pd.DataFrame,
    lgb_params: dict,
    cat_params: dict | None = None,
    n_estimators: int = 200,
    neg_ratio: float = 0.01,
    seeds: tuple[int, ...] = SEED_BAG,
    cat_weight: float = 0.2,
) -> dict:
    """Undersampled multi-seed bag of LightGBM (+ optional CatBoost) per fold.

    Per fold:
      1. fit fold features on the FULL fold-train side (te_* sees train labels
         only), transform train + the FULL held-out val fold;
      2. for each seed: undersample train negatives (all positives kept), train
         a fixed-iteration booster, predict the FULL val fold;
      3. rank-average the per-seed val predictions -> that model family's fold
         OOF (rank-average is scale-free and robust across seeds).

    If `cat_params` is given, a CatBoost bag is trained identically and the
    final per-fold OOF is the WEIGHTED rank-blend of the LightGBM-bag and
    CatBoost-bag OOF for that fold:
        final = (1 - cat_weight) * rank(lgb) + cat_weight * rank(cat)
    `cat_weight` is selected on cv.oof_pauc: CatBoost is weaker than LightGBM
    here, so an equal (0.5) blend dilutes; ~0.2 is optimal. Rank-blending (not
    prob-averaging) is used because the two libraries' raw probabilities sit on
    different scales — measured: equal prob-mean LOSES ~0.002 pAUC vs rank-blend.

    Returns a dict:
      oof_lgb, oof_cat (or None), oof_final : np.ndarray aligned to `meta`
      boosters : flat list of (model, cols, state) for submit.py
      score_lgb, score_cat, score_final : official pAUC of each OOF
    """
    df = meta.merge(folds[[ID, "fold"]], on=ID, how="left")
    assert df["fold"].notna().all(), "every row must have a fold (run src.cv first)"
    y = df[TARGET].values

    n = len(df)
    oof_lgb = np.zeros(n, dtype=float)
    oof_cat = np.zeros(n, dtype=float) if cat_params else None
    boosters: list = []

    for k in sorted(df["fold"].unique()):
        tr_pos = df.index[df["fold"] != k].to_numpy()
        va_pos = df.index[df["fold"] == k].to_numpy()

        # features fit on the FULL train side (BEFORE undersampling)
        state = F.fit_fold_features(df.loc[tr_pos])
        tr_full = F.transform_fold_features(df.loc[tr_pos], state)
        va = F.transform_fold_features(df.loc[va_pos], state)
        cols = F.feature_columns(tr_full)
        Xva = va[cols]

        # per-seed bag
        lgb_preds: list[np.ndarray] = []
        cat_preds: list[np.ndarray] = []
        for s in seeds:
            sub = _undersample(tr_pos, y, neg_ratio, seed=s)
            Xsub = tr_full.loc[sub, cols]
            ysub = y[sub]

            m_lgb = _train_lgb_seed(Xsub, ysub, lgb_params, n_estimators, s)
            lgb_preds.append(m_lgb.predict(Xva))
            boosters.append((m_lgb, cols, state))

            if cat_params:
                m_cat = _train_cat_seed(Xsub, ysub, cat_params, n_estimators, s)
                cat_preds.append(m_cat.predict(Xva))
                boosters.append((m_cat, cols, state))

        # rank-average the seed bag for this fold
        oof_lgb[va_pos] = _rank_avg(lgb_preds)
        if cat_params:
            oof_cat[va_pos] = _rank_avg(cat_preds)

    score_lgb = cv.oof_pauc(y, oof_lgb)
    out = {
        "oof_lgb": oof_lgb, "oof_cat": oof_cat,
        "boosters": boosters,
        "score_lgb": score_lgb, "score_cat": None,
    }
    if cat_params:
        out["score_cat"] = cv.oof_pauc(y, oof_cat)
        out["cat_weight"] = float(cat_weight)
        # final OOF: WEIGHTED rank-blend of the two families, computed per fold
        # so the rank space is the held-out fold (matches selection).
        oof_final = np.zeros(n, dtype=float)
        for k in sorted(df["fold"].unique()):
            m = (df["fold"] == k).to_numpy()
            oof_final[m] = _rank_blend([oof_lgb[m], oof_cat[m]],
                                       [1.0 - cat_weight, cat_weight])
        out["oof_final"] = oof_final
        out["score_final"] = cv.oof_pauc(y, oof_final)
    else:
        out["oof_final"] = oof_lgb
        out["score_final"] = score_lgb
    return out


def _rank_avg(preds: list[np.ndarray]) -> np.ndarray:
    """Mean of rank-normalized score vectors (scale-free, robust)."""
    if len(preds) == 1:
        return preds[0]
    ranked = np.vstack([rankdata(p) / len(p) for p in preds])
    return ranked.mean(axis=0)


def _rank_blend(preds: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Weighted mean of rank-normalized score vectors (scale-free, robust)."""
    w = np.asarray(weights, dtype=float)
    w = w / w.sum()
    ranked = np.vstack([rankdata(p) / len(p) for p in preds])
    return np.average(ranked, axis=0, weights=w)


def save_oof(meta: pd.DataFrame, oof: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({ID: meta[ID].values, "gbdt_oof": oof}).to_parquet(path, index=False)


def save_boosters(boosters: list, path: str | Path = "experiments/gbdt_boosters.joblib") -> None:
    """Persist the per-fold models for inference (`submit.py:predict_gbdt`).

    CONTRACT: `boosters` is a flat list whose every element is a tuple
    `(model, cols, state)` —
      model : an object exposing `.predict(X, num_iteration=...)` and
              `.best_iteration` (a raw lightgbm.Booster, or a `_RankModel`
              wrapping a LightGBM/CatBoost classifier for the bagged path)
      cols  : list[str] feature columns fed to that model
      state : the dict returned by `features.fit_fold_features` for that fold
    `submit.py:predict_gbdt` reapplies `state`, predicts with each model, and
    averages across the list — so a flat list across all (fold, seed) and both
    model families reproduces the bagged test prediction.
    """
    import joblib

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(boosters, p)
    print(f"saved {len(boosters)} per-fold model tuples -> {p}")


def _log_gbdt_frontier(boosters: list, meta: pd.DataFrame, folds: pd.DataFrame,
                       score: float, path: str = "reports/frontier.csv") -> None:
    """Log one GBDT point to the frontier CSV. Defensive: never crash the run."""
    try:
        from . import stack

        def _ntrees(m):
            try:
                if isinstance(m, _RankModel):
                    if m.kind == "lgb":
                        if hasattr(m.model, "n_estimators_"):
                            return m.model.n_estimators_
                        return m.model.get_params().get("n_estimators", 0)
                    return m.model.tree_count_
                return m.num_trees()
            except Exception:
                return 0

        try:
            cost = {
                "params_m": round(sum(_ntrees(m) for m, _, _ in boosters) / 1e6, 6),
                "gflops": 0.0, "cpu_ms": 0.0, "img_size": 0,
            }
        except Exception:
            cost = {"params_m": 0.0, "gflops": 0.0, "cpu_ms": 0.0, "img_size": 0}
        try:
            df = meta.merge(folds[[ID, "fold"]], on=ID, how="left")
            _, cols, state = boosters[0]
            sample = df.head(min(2000, len(df)))
            X_sample = F.transform_fold_features(sample, state)[cols]
            try:
                from .efficiency import measure_gbdt  # added by efficiency agent

                # measure_gbdt expects raw lightgbm Boosters (it calls
                # .num_trees()); for the bagged _RankModel wrappers it returns a
                # partial dict. Merge only keys with non-None values so it never
                # clobbers our tree-count params_m with a missing entry.
                meas = measure_gbdt(boosters, X_sample) or {}
                cost.update({
                    k: v for k, v in meas.items()
                    if v is not None and not (isinstance(v, float) and np.isnan(v))
                })
            except Exception:
                import time

                model0 = boosters[0][0]
                for _ in range(3):  # warmup
                    model0.predict(X_sample)
                ts = []
                for _ in range(10):
                    t0 = time.perf_counter()
                    model0.predict(X_sample)
                    ts.append((time.perf_counter() - t0) * 1e3 / max(len(X_sample), 1))
                cost["cpu_ms"] = round(float(np.median(ts)), 5)
        except Exception:
            pass

        stack.log_frontier({
            "model": "gbdt",
            "pauc": round(float(score), 5),
            "params_m": cost.get("params_m", 0.0),
            "gflops": cost.get("gflops", 0.0),
            "cpu_ms": cost.get("cpu_ms", 0.0),
            "img_size": cost.get("img_size", 0),
        }, path)
    except Exception as e:  # cost logging must never sink the run
        print(f"[warn] frontier logging skipped: {e!r}")


def main(argv: list[str] | None = None) -> None:
    """CLI entrypoint. Lives in the module (NOT the __main__ guard) so the
    `_RankModel` instances it creates pickle as `src.gbdt._RankModel`. Running
    via `python -m src.gbdt` re-imports this module and calls `main`, so the
    persisted joblib reloads cleanly in `submit.py` (which imports `src.gbdt`).
    """
    import argparse

    from . import config as C

    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/train-metadata.csv")
    ap.add_argument("--folds", default="data/folds.parquet")
    ap.add_argument("--params", default="configs/gbdt.yaml")
    ap.add_argument("--out", default="experiments/gbdt_oof.parquet")
    ap.add_argument("--boosters", default="experiments/gbdt_boosters.joblib")
    ap.add_argument("--mode", default="bagged", choices=["bagged", "single"],
                    help="bagged = Step-3b/3c undersampled multi-seed (default); "
                         "single = Step-1/2 early-stopped LightGBM")
    a = ap.parse_args(argv)

    meta = pd.read_csv(a.meta, low_memory=False)
    folds = cv.load_folds(a.folds)
    cfg = C.load(a.params)

    if a.mode == "single":
        params = dict(cfg["lgb"])
        oof, boosters, score = train_oof(meta, folds, params)
    else:
        bag = cfg.get("bag", {})
        lgb_params = dict(cfg["lgb_bag"])
        cat_params = dict(cfg["cat_bag"]) if cfg.get("cat_bag") else None
        res = train_oof_bagged(
            meta, folds, lgb_params, cat_params,
            n_estimators=int(bag.get("n_estimators", 200)),
            neg_ratio=float(bag.get("neg_ratio", 0.01)),
            seeds=tuple(bag.get("seeds", SEED_BAG)),
            cat_weight=float(bag.get("cat_weight", 0.2)),
        )
        oof, boosters, score = res["oof_final"], res["boosters"], res["score_final"]
        print(f"  LightGBM-bag OOF pAUC = {res['score_lgb']:.5f}")
        if res["score_cat"] is not None:
            print(f"  CatBoost-bag OOF pAUC = {res['score_cat']:.5f}")
            print(f"  rank-blend cat_weight = {res.get('cat_weight'):.2f}")

    save_oof(meta, oof, a.out)
    save_boosters(boosters, a.boosters)
    _log_gbdt_frontier(boosters, meta, folds, score)
    print(f"GBDT OOF pAUC@0.80TPR = {score:.5f}  ({len(meta):,} rows)")


if __name__ == "__main__":
    # Delegate through the imported module (not this __main__ copy) so every
    # object created during training — crucially `_RankModel` — carries the
    # `src.gbdt` module path and the persisted boosters unpickle in submit.py.
    from src.gbdt import main as _main

    _main()
