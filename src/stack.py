"""Combiner + frontier logging.

Two stacking strategies, both deliberately simple (the 393-positive regime
punishes complexity):
  - 'rank'  : average of per-model rank-normalized OOF scores (robust, param-free)
  - 'gbdt'  : LightGBM meta-learner over the experts' OOF + image embeddings

The learned per-lesion gate ("MoE-style") lives in `gate_ablation` and is run
ONLY to report whether it beats the trivial combiner. It usually won't here;
that is a result, not a failure.

Every combination is scored with the official metric and logged to the frontier
CSV alongside the summed cost of its constituent models.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from . import cv
from .data import GROUP, ID, TARGET


def rank_average(oofs: list[np.ndarray]) -> np.ndarray:
    return np.mean([rankdata(o) / len(o) for o in oofs], axis=0)


def assemble_feature_frame(oof_paths: list[str]) -> pd.DataFrame:
    """Join the listed OOF parquets on `isic_id` into one feature frame.

    Each path is one expert's OOF parquet following the data contract:
      - GBDT   : columns [isic_id, gbdt_oof]
      - vision : columns [isic_id, {name}_oof, {name}_emb0, {name}_emb1, ...]

    The result keeps `isic_id` plus every non-id column from every file, and is
    ready to hand to `gbdt_stack`'s `feature_frame` arg. An inner join on the id
    is used so the stack only sees rows every expert scored (defensive: experts
    on the same frozen folds cover the same rows, but a partial OOF — e.g. a
    vision run that crashed mid-fold — must not silently inject NaNs into the
    meta-learner). Duplicate non-id columns across files raise rather than
    clobber, so a contract mismatch is loud.
    """
    if not oof_paths:
        raise ValueError("assemble_feature_frame: no OOF paths given")

    frame: pd.DataFrame | None = None
    seen: set[str] = set()
    for path in oof_paths:
        df = pd.read_parquet(path)
        if ID not in df.columns:
            raise ValueError(f"{path}: missing {ID!r} column (broken OOF contract)")
        feat_cols = [c for c in df.columns if c != ID]
        clash = seen.intersection(feat_cols)
        if clash:
            raise ValueError(
                f"{path}: duplicate feature column(s) {sorted(clash)} "
                "already supplied by an earlier OOF file"
            )
        seen.update(feat_cols)
        df = df[[ID] + feat_cols]
        frame = df if frame is None else frame.merge(df, on=ID, how="inner")

    return frame.reset_index(drop=True)


def image_ugly_duckling(oof_parquet_path: str, name: str,
                        folds_path: str = "data/folds.parquet",
                        eps: float = 1e-6) -> pd.DataFrame:
    """Image-space "ugly-duckling": per-patient embedding deviation features.

    The classic ugly-duckling sign in dermatology: the lesion that does not look
    like the patient's other lesions is the suspicious one. We port that intuition
    into image-embedding space. For each lesion's embedding x_i belonging to
    patient p, with the patient's own embeddings forming a cloud, we compute how
    far x_i sits from that cloud:

      - {name}_imgdev_eucl : Euclidean distance to the patient centroid mu_p.
      - {name}_imgdev_cos  : cosine distance (1 - cos sim) to mu_p.
      - {name}_imgdev_zl2  : L2 norm of the per-dimension z-scores,
                             (x_i - mu_p) / (sigma_p + eps), i.e. a
                             whitened/Mahalanobis-lite deviation.

    Leak-safety: everything is computed WITHIN a patient (groupby patient_id) and
    NEVER touches the target. Because the frozen folds keep every patient in
    exactly one fold, a patient-level statistic cannot carry information across
    the fold boundary -- the centroid for fold-k patients is built only from
    fold-k rows. So these features are safe to drop straight into OOF stacking.

    Memory note: the embedding matrix is ~640 x 401k. We read only the embedding
    columns, keep them float32, accumulate per-patient mean/var with vectorized
    groupby transforms, and delete intermediates as we go (this box has 31GB).

    Returns a frame [isic_id, {name}_imgdev_eucl, {name}_imgdev_cos,
    {name}_imgdev_zl2].
    """
    import pyarrow.parquet as pq

    schema_names = pq.read_schema(oof_parquet_path).names
    emb_cols = [c for c in schema_names if c.startswith(f"{name}_emb")]
    if not emb_cols:
        raise ValueError(
            f"image_ugly_duckling: no embedding columns '{name}_emb*' in "
            f"{oof_parquet_path}"
        )

    df = pd.read_parquet(oof_parquet_path, columns=[ID] + emb_cols)
    ids = df[ID].to_numpy()
    X = df[emb_cols].to_numpy(dtype=np.float32)
    del df

    # patient_id comes from the frozen fold table (leak-safe grouping key); align
    # it to the embedding rows by isic_id.
    folds = pd.read_parquet(folds_path, columns=[ID, GROUP])
    pid = (pd.DataFrame({ID: ids})
           .merge(folds, on=ID, how="left")[GROUP].to_numpy())
    del folds
    if pd.isna(pid).any():
        raise ValueError(
            "image_ugly_duckling: some isic_ids have no patient_id in the fold "
            "table -- refusing to compute deviations against an unknown cloud."
        )

    # Group rows by patient via a single argsort, then process contiguous blocks.
    # This bounds peak memory: at any time we hold X (the embeddings) plus a few
    # patient-sized scratch arrays.
    order = np.argsort(pid, kind="stable")
    pid_sorted = pid[order]
    # boundaries of contiguous patient blocks in the sorted order
    boundaries = np.flatnonzero(np.r_[True, pid_sorted[1:] != pid_sorted[:-1]])
    starts = boundaries
    ends = np.r_[boundaries[1:], len(pid_sorted)]

    n = len(ids)
    eucl = np.empty(n, dtype=np.float32)
    cosd = np.empty(n, dtype=np.float32)
    zl2 = np.empty(n, dtype=np.float32)

    for s, e in zip(starts, ends):
        idx = order[s:e]                      # original-row indices for this patient
        blk = X[idx]                          # (m, d) float32 view-copy
        m = blk.shape[0]
        mu = blk.mean(axis=0)                 # patient centroid (d,)
        diff = blk - mu                       # (m, d)

        eucl[idx] = np.sqrt(np.einsum("ij,ij->i", diff, diff))

        # cosine distance to centroid
        mu_norm = float(np.sqrt(mu @ mu))
        if m == 1 or mu_norm < eps:
            # single-lesion patient (no cloud) or degenerate centroid -> 0 deviation
            cosd[idx] = 0.0
        else:
            blk_norm = np.sqrt(np.einsum("ij,ij->i", blk, blk))
            denom = blk_norm * mu_norm
            sim = (blk @ mu) / np.where(denom < eps, 1.0, denom)
            cosd[idx] = 1.0 - sim

        # L2 of per-dim z-scores (population std within patient)
        if m == 1:
            zl2[idx] = 0.0
        else:
            sigma = blk.std(axis=0)           # (d,) population std (ddof=0)
            z = diff / (sigma + eps)
            zl2[idx] = np.sqrt(np.einsum("ij,ij->i", z, z))
            del sigma, z
        del blk, diff

    del X

    return pd.DataFrame({
        ID: ids,
        f"{name}_imgdev_eucl": eucl,
        f"{name}_imgdev_cos": cosd,
        f"{name}_imgdev_zl2": zl2,
    })


def gbdt_stack(meta: pd.DataFrame, folds: pd.DataFrame, feature_frame: pd.DataFrame,
               params: dict) -> tuple[np.ndarray, float]:
    """Meta-LightGBM over expert OOFs (+ embeddings), trained on the same folds."""
    import lightgbm as lgb

    df = (meta[[ID, TARGET]]
          .merge(folds[[ID, "fold"]], on=ID)
          .merge(feature_frame, on=ID))
    cols = [c for c in feature_frame.columns if c != ID]
    y = df[TARGET].values
    oof = np.zeros(len(df))
    for k in sorted(df["fold"].unique()):
        tr = df[df["fold"] != k]
        va = df[df["fold"] == k]
        booster = lgb.train(
            {**params, "objective": "binary", "verbosity": -1},
            lgb.Dataset(tr[cols], label=tr[TARGET]),
            num_boost_round=400,
        )
        oof[df.index[df["fold"] == k]] = booster.predict(va[cols])
    return oof, cv.oof_pauc(y, oof)


def _to_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Map probabilities (or any bounded score) to a logit-like real line.

    If values already look like logits (outside [0, 1]) they are passed through
    after standardization downstream; otherwise clip to (0, 1) and apply logit.
    """
    p = np.asarray(p, dtype=float).ravel()
    if p.min() >= 0.0 and p.max() <= 1.0:
        p = np.clip(p, eps, 1.0 - eps)
        return np.log(p / (1.0 - p))
    return p


def gate_ablation(meta: pd.DataFrame, folds: pd.DataFrame,
                  tab_logit_col: str, img_logit_col: str,
                  feature_frame: pd.DataFrame, params: dict | None = None
                  ) -> tuple[np.ndarray, float]:
    """Honest MoE negative-result: a tiny learned per-lesion softmax gate.

    A minimal-capacity gate predicts, per lesion, a 2-way softmax weight over the
    [tabular_logit, image_logit] experts from a few cheap context features, then
    blends the two expert logits with those weights. The gate is a heavily L2-
    regularized logistic model (one set of weights -> a single softmax over the
    two experts), trained PER FOLD on the frozen folds so the OOF blend is
    leak-free, and scored with the official metric.

    Expectation (CLAUDE.md): with 393 positives this does NOT beat
    `rank_average`. That is the point — log both as frontier rows.

    Args:
        meta: metadata frame (provides isic_id + target).
        folds: frozen fold table from cv.load_folds.
        tab_logit_col: column in `feature_frame` holding the tabular expert score
            (prob or logit; converted to logit internally).
        img_logit_col: column in `feature_frame` holding the image expert score.
        feature_frame: joined expert OOFs/embeddings (see assemble_feature_frame).
            Embedding columns, if present, become the gate's context features.
        params: optional overrides, e.g. {"l2": 100.0, "max_iter": 200}.

    Returns:
        (oof, pauc): the OOF blended score aligned to the merged frame, and its
        official pAUC@80%TPR.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    p = {"l2": 100.0, "max_iter": 500}
    if params:
        p.update(params)
    # heavy L2 == tiny capacity: sklearn's C is the inverse regularization strength.
    C = 1.0 / float(p["l2"])

    df = (meta[[ID, TARGET]]
          .merge(folds[[ID, "fold"]], on=ID)
          .merge(feature_frame, on=ID)
          .reset_index(drop=True))
    for col in (tab_logit_col, img_logit_col):
        if col not in df.columns:
            raise ValueError(f"gate_ablation: column {col!r} not in feature_frame")

    y = df[TARGET].values.astype(int)
    tab_logit = _to_logit(df[tab_logit_col].values)
    img_logit = _to_logit(df[img_logit_col].values)

    # Context features the gate routes on: any embeddings present + the two expert
    # logits themselves. Deliberately few, so capacity stays minimal.
    ctx_cols = [c for c in df.columns
                if c not in {ID, TARGET, "fold"}
                and pd.api.types.is_numeric_dtype(df[c])]
    X_ctx = df[ctx_cols].to_numpy(dtype=float)
    X_ctx = np.nan_to_num(X_ctx, nan=0.0, posinf=0.0, neginf=0.0)

    oof = np.zeros(len(df), dtype=float)
    for k in sorted(df["fold"].unique()):
        tr = (df["fold"] != k).to_numpy()
        va = (df["fold"] == k).to_numpy()

        scaler = StandardScaler().fit(X_ctx[tr])
        Xtr = scaler.transform(X_ctx[tr])
        Xva = scaler.transform(X_ctx[va])

        # The gate target: which expert is "more right" for each TRAIN lesion?
        # We frame the per-lesion routing as a binary problem — choose the image
        # expert (label 1) when it gives the more correct logit for that lesion's
        # true label, else the tabular expert (label 0). The logistic model then
        # predicts a soft routing weight (its positive-class probability) which we
        # use as the image-expert mixing weight in a 2-way softmax blend.
        sign = np.where(y[tr] == 1, 1.0, -1.0)            # reward higher logit on pos
        img_better = (img_logit[tr] * sign) > (tab_logit[tr] * sign)
        gate_y = img_better.astype(int)

        if len(np.unique(gate_y)) < 2:
            # degenerate fold: one expert always wins -> fall back to even blend.
            w_img = np.full(va.sum(), 0.5)
        else:
            clf = LogisticRegression(C=C, max_iter=int(p["max_iter"]),
                                     class_weight="balanced")
            clf.fit(Xtr, gate_y)
            w_img = clf.predict_proba(Xva)[:, 1]

        # 2-way softmax over [tabular, image] reduces to the scalar mixing weight.
        oof[va] = (1.0 - w_img) * tab_logit[va] + w_img * img_logit[va]

    return oof, cv.oof_pauc(y, oof)


def evaluate_and_log(name: str, oof: np.ndarray, y: np.ndarray, cost_row: dict,
                     frontier_path: str | Path = "reports/frontier.csv") -> float:
    """Score `oof` with the official metric and append a frontier row.

    cost_row supplies the cost columns (params_m, gflops, cpu_ms, img_size) for
    a combiner this is typically the summed/maxed cost of its constituents (or
    zeros, since the combiner itself is ~free). Returns the pAUC.
    """
    pauc = cv.oof_pauc(np.asarray(y), np.asarray(oof))
    log_frontier({"model": name, "pauc": pauc, **cost_row}, frontier_path)
    return pauc


def log_frontier(row: dict, path: str | Path = "reports/frontier.csv") -> None:
    """Append one (model, pAUC, params, FLOPs, latency) point to the frontier."""
    p = Path(path)
    df = pd.DataFrame([row])
    if p.exists():
        df = pd.concat([pd.read_csv(p), df], ignore_index=True)
    p.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(p, index=False)
