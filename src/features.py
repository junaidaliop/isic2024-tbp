"""Intrinsic tabular feature engineering for SLICE-3D (ISIC-2024 metadata only).

Leak-safety: target encodings are fit per-fold on the train side in
`fit_fold_features` and only looked up in `transform_fold_features`; patient-level
aggregates use no target and are computed within each frame.
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

from .data import GROUP, ID, TARGET

# Ablation toggle (not a leak switch): 0 reproduces the lean Step-1 baseline.
WIDE_DEV = os.environ.get("ISIC_FE_WIDE_DEV", "1") != "0"

SIZE_COLS = ["clin_size_long_diam_mm", "tbp_lv_areaMM2", "tbp_lv_perimeterMM",
             "tbp_lv_minorAxisMM"]
COLOR_COLS = ["tbp_lv_A", "tbp_lv_B", "tbp_lv_L", "tbp_lv_Aext", "tbp_lv_Bext",
              "tbp_lv_Lext", "tbp_lv_color_std_mean", "tbp_lv_deltaA",
              "tbp_lv_deltaB", "tbp_lv_deltaL", "tbp_lv_deltaLBnorm"]
SHAPE_COLS = ["tbp_lv_eccentricity", "tbp_lv_symm_2axis", "tbp_lv_symm_2axis_angle",
              "tbp_lv_norm_border", "tbp_lv_norm_color"]

# Categoricals to per-fold target-encode (each guarded by presence check).
CAT_COLS = ["anatom_site_general", "sex", "tbp_lv_location",
            "tbp_lv_location_simple"]

# Heavy smoothing: a category needs ~TE_SMOOTHING train rows before its own mean
# outweighs the prior — matters in the 393-positive regime.
TE_SMOOTHING = 50.0

# Train-only / post-hoc-diagnosis columns absent from the public test metadata;
# they leak the label or are unavailable at inference. Several are float64 and
# would pass a numeric-dtype filter, so drop them explicitly.
LEAK_COLS = {
    "lesion_id", "iddx_full", "iddx_1", "iddx_2", "iddx_3", "iddx_4", "iddx_5",
    "mel_mitotic_index", "mel_thick_mm", "tbp_lv_dnn_lesion_confidence",
}


# Engineered ratio/contrast columns also patient-normalized by the wide
# deviation generator; keep in sync with names produced in `base_features`.
ENG_DEV_COLS = [
    "f_shape_index", "f_axis_ratio", "f_lesion_size_ratio", "f_lesion_shape_index",
    "f_area_to_perimeter", "f_normalized_lesion_size", "f_color_contrast",
    "f_lesion_color_difference", "f_hue_contrast", "f_luminance_contrast",
    "f_color_uniformity", "f_border_complexity", "f_3d_position_distance",
    "f_irregularity",
]

# Step-3a patient x category deviation toggle (not a leak switch); 0 reproduces
# the Step-2 feature set.
PXC = os.environ.get("ISIC_FE_PXC", "1") != "0"

# Per-lesion categoricals to slice the within-patient deviation by. sex is
# excluded: patient-constant, so groupby(patient, sex) == groupby(patient).
PXC_CAT_COLS = ["anatom_site_general", "tbp_lv_location", "tbp_lv_location_simple"]

# Curated numeric bases for the patient x category deviation (capped for
# parsimony: 25 numerics x 3 categoricals = 75 pxc_ features). All target-free.
PXC_NUM_COLS = [
    # size / geometry
    "clin_size_long_diam_mm", "tbp_lv_areaMM2", "tbp_lv_perimeterMM",
    "tbp_lv_minorAxisMM", "tbp_lv_area_perim_ratio",
    # shape / border irregularity
    "tbp_lv_eccentricity", "tbp_lv_symm_2axis", "tbp_lv_norm_border",
    "tbp_lv_norm_color",
    # colour vs surrounding skin
    "tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL", "tbp_lv_deltaLBnorm",
    "tbp_lv_color_std_mean", "tbp_lv_radial_color_std_max", "tbp_lv_stdL",
    # absolute LAB lesion colour
    "tbp_lv_L", "tbp_lv_A", "tbp_lv_B", "tbp_lv_C", "tbp_lv_H",
    "tbp_lv_nevi_confidence",
    # engineered composites
    "f_color_contrast", "f_lesion_color_difference", "f_border_complexity",
]


def base_features(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless, per-row engineered features (safe; no cross-row leakage)."""
    out = df.copy()
    eps = 1e-6

    # --- geometry / size ratios ------------------------------------------
    if {"tbp_lv_perimeterMM", "tbp_lv_areaMM2"}.issubset(out.columns):
        out["f_shape_index"] = out["tbp_lv_perimeterMM"] / (np.sqrt(out["tbp_lv_areaMM2"]) + eps)
    if {"clin_size_long_diam_mm", "tbp_lv_minorAxisMM"}.issubset(out.columns):
        out["f_axis_ratio"] = out["clin_size_long_diam_mm"] / (out["tbp_lv_minorAxisMM"] + eps)

    # lesion-vs-surround color contrast (the "ugly duckling" colour signal)
    for c in ["tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL", "tbp_lv_deltaLBnorm"]:
        if c in out.columns:
            out[f"f_abs_{c}"] = out[c].abs()

    # aggregate colour-contrast magnitude across the LAB delta channels
    delta_abs = [f"f_abs_{c}" for c in ["tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL"]
                 if f"f_abs_{c}" in out.columns]
    if delta_abs:
        out["f_color_contrast"] = out[delta_abs].sum(axis=1)

    # border / color irregularity composite
    if {"tbp_lv_norm_border", "tbp_lv_norm_color"}.issubset(out.columns):
        out["f_irregularity"] = out["tbp_lv_norm_border"] * out["tbp_lv_norm_color"]

    # --- Step-2 expanded geometry / colour / position --------------------
    # (gated by WIDE_DEV so the Step-1 baseline can be reproduced for ablation)
    if WIDE_DEV:
        # long-vs-minor axis ratio (elongation); kept distinct from f_axis_ratio
        if {"clin_size_long_diam_mm", "tbp_lv_minorAxisMM"}.issubset(out.columns):
            out["f_lesion_size_ratio"] = (
                out["clin_size_long_diam_mm"] / (out["tbp_lv_minorAxisMM"] + eps))
        # isoperimetric-style compactness and area/perimeter
        if {"tbp_lv_areaMM2", "tbp_lv_perimeterMM"}.issubset(out.columns):
            out["f_lesion_shape_index"] = (
                out["tbp_lv_areaMM2"] / (out["tbp_lv_perimeterMM"] ** 2 + eps))
            out["f_area_to_perimeter"] = out["tbp_lv_areaMM2"] / (out["tbp_lv_perimeterMM"] + eps)
        # size normalized by age (faster-growing relative to lifetime is suspicious)
        if {"clin_size_long_diam_mm", "age_approx"}.issubset(out.columns):
            out["f_normalized_lesion_size"] = (
                out["clin_size_long_diam_mm"] / (out["age_approx"] + eps))
        # Euclidean LAB distance between lesion and surrounding skin
        if {"tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL"}.issubset(out.columns):
            out["f_lesion_color_difference"] = np.sqrt(
                out["tbp_lv_deltaA"] ** 2 + out["tbp_lv_deltaB"] ** 2 + out["tbp_lv_deltaL"] ** 2
            )
        # hue / luminance contrast vs surrounding skin
        if {"tbp_lv_H", "tbp_lv_Hext"}.issubset(out.columns):
            out["f_hue_contrast"] = (out["tbp_lv_H"] - out["tbp_lv_Hext"]).abs()
        if {"tbp_lv_L", "tbp_lv_Lext"}.issubset(out.columns):
            out["f_luminance_contrast"] = (out["tbp_lv_L"] - out["tbp_lv_Lext"]).abs()
        # internal colour heterogeneity normalized by radial colour spread
        if {"tbp_lv_color_std_mean", "tbp_lv_radial_color_std_max"}.issubset(out.columns):
            out["f_color_uniformity"] = (
                out["tbp_lv_color_std_mean"] / (out["tbp_lv_radial_color_std_max"] + eps))
        # border roughness + asymmetry combined into one complexity score
        if {"tbp_lv_norm_border", "tbp_lv_symm_2axis"}.issubset(out.columns):
            out["f_border_complexity"] = out["tbp_lv_norm_border"] + out["tbp_lv_symm_2axis"]
        # 3D body-surface position magnitude
        if {"tbp_lv_x", "tbp_lv_y", "tbp_lv_z"}.issubset(out.columns):
            out["f_3d_position_distance"] = np.sqrt(
                out["tbp_lv_x"] ** 2 + out["tbp_lv_y"] ** 2 + out["tbp_lv_z"] ** 2
            )

    # --- modest interaction terms ----------------------------------------
    # irregular shape that is also a borderline/colour outlier is the classic
    # malignant signature; multiply so the model gets the joint cheaply.
    if {"f_shape_index", "f_irregularity"}.issubset(out.columns):
        out["f_shape_x_irreg"] = out["f_shape_index"] * out["f_irregularity"]
    # a big lesion with high colour contrast is more suspicious than either alone
    if {"f_color_contrast", "clin_size_long_diam_mm"}.issubset(out.columns):
        out["f_contrast_x_size"] = out["f_color_contrast"] * out["clin_size_long_diam_mm"]
    # eccentric + asymmetric: two independent shape-irregularity axes interacting
    if {"tbp_lv_eccentricity", "tbp_lv_symm_2axis"}.issubset(out.columns):
        out["f_ecc_x_asymm"] = out["tbp_lv_eccentricity"] * (1.0 - out["tbp_lv_symm_2axis"])

    return out


def _patient_features(out: pd.DataFrame) -> pd.DataFrame:
    """Patient-level count + within-patient size rank. Target-free, leak-safe.

    Computed within whatever frame it is handed (train fold, val fold, or test),
    so it never reads across the fold boundary. No label is used.
    """
    if GROUP not in out.columns:
        return out

    g = out.groupby(GROUP)
    # how many lesions this patient contributes (context for ugly-duckling logic)
    out["patient_n_lesions"] = g[GROUP].transform("size").astype(float)

    # size rank of this lesion within the patient, normalized to [0, 1]
    size_col = "clin_size_long_diam_mm" if "clin_size_long_diam_mm" in out.columns else None
    if size_col is None:
        for c in SIZE_COLS:
            if c in out.columns:
                size_col = c
                break
    if size_col is not None:
        # average-rank handles ties; normalize by patient lesion count so the
        # feature is comparable across patients of different sizes
        ranks = g[size_col].rank(method="average")
        denom = out["patient_n_lesions"].clip(lower=1.0)
        out["f_size_rank_in_patient"] = ((ranks - 1.0) / denom).astype(float)

    return out


def _deviation_base_cols(out: pd.DataFrame) -> list[str]:
    """The numeric base+engineered columns to patient-normalize ("ugly duckling").

    Every raw ``tbp_lv_*`` numeric column present (the TBP-derived measurements),
    plus the key engineered size/colour ratios and ``clin_size_long_diam_mm``.
    LEAK_COLS are never in this set (none of them start with the kept prefixes and
    they are filtered defensively anyway), so this stays leak-safe.
    """
    import pandas.api.types as pdt

    cols: list[str] = []
    for c in out.columns:
        if c in LEAK_COLS:
            continue
        keep = c.startswith("tbp_lv_") or c == "clin_size_long_diam_mm" or c in ENG_DEV_COLS
        if keep and pdt.is_numeric_dtype(out[c]):
            cols.append(c)
    return cols


def _patient_deviations(out: pd.DataFrame) -> pd.DataFrame:
    """Wide within-patient z-score + min-max rank deviations. Target-free, leak-safe.

    For each base feature x and patient p:
      pdev_{x} = (x - mean_p(x)) / (std_p(x) + eps)   -- standardized deviation
      prank_{x} = (x - min_p(x)) / (max_p(x) - min_p(x) + eps)  -- [0,1] position
    Both are computed within whatever frame is passed (train / val / test each in
    isolation); patients never straddle a fold, so no cross-fold leakage. No label.
    Single-lesion patients get std 0 -> pdev 0, and a degenerate range -> prank 0.
    """
    if GROUP not in out.columns:
        return out

    eps = 1e-6
    cols = _deviation_base_cols(out)
    if not cols:
        return out

    g = out.groupby(GROUP)
    new_cols: dict[str, np.ndarray] = {}
    for c in cols:
        gc = g[c]
        mean = gc.transform("mean")
        std = gc.transform("std")
        new_cols[f"pdev_{c}"] = ((out[c] - mean) / (std + eps)).to_numpy()
        cmin = gc.transform("min")
        cmax = gc.transform("max")
        new_cols[f"prank_{c}"] = ((out[c] - cmin) / (cmax - cmin + eps)).to_numpy()

    # concat once (avoids fragmented-frame perf warnings on ~hundreds of columns)
    out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)
    return out


def _patient_category_deviations(out: pd.DataFrame) -> pd.DataFrame:
    """Step-3a PATIENT x CATEGORY "ugly-duckling-at-the-same-site" deviations.

    For each curated numeric base column x, each per-lesion categorical cat, and
    each (patient p, category value c):

        pxc_{x}_{cat} = (x - mean_{p,c}(x)) / (std_{p,c}(x) + eps)

    i.e. how far this lesion deviates from the SAME patient's OTHER lesions AT
    THE SAME body site / location. This is greysky's differentiator: a 6mm mole
    is unremarkable on someone whose torso lesions are all ~6mm, but a striking
    outlier if it is the only large one at that site.

    Leak-safety: no target is touched, and the groupby is keyed on patient_id, so
    every group lives entirely inside one fold (patients never straddle the frozen
    folds). Computed within whatever frame is passed (train / val / test each in
    isolation). Singleton (patient, site) groups get std 0 -> deviation 0.
    Missing category labels are filled with a sentinel so NaN-keyed groups don't
    silently merge distinct lesions.
    """
    if not PXC or GROUP not in out.columns:
        return out

    eps = 1e-6
    cats = [c for c in PXC_CAT_COLS if c in out.columns]
    nums = [c for c in PXC_NUM_COLS if c in out.columns]
    if not cats or not nums:
        return out

    new_cols: dict[str, np.ndarray] = {}
    for cat in cats:
        # sentinel-fill so missing site labels form their own group rather than
        # being dropped (pandas groupby drops NaN keys by default)
        cat_key = out[cat].astype(str).fillna("__NA__")
        g = out.groupby([out[GROUP], cat_key])
        for x in nums:
            gx = g[x]
            mean = gx.transform("mean")
            std = gx.transform("std")
            new_cols[f"pxc_{x}_{cat}"] = ((out[x] - mean) / (std + eps)).to_numpy()

    # concat once to avoid fragmented-frame perf warnings on ~75 columns
    out = pd.concat([out, pd.DataFrame(new_cols, index=out.index)], axis=1)
    return out


def fit_fold_features(train: pd.DataFrame) -> dict:
    """Learn per-fold statistics on the TRAIN side only (leak-free).

    Returns a state dict consumed by `transform_fold_features`. Patient-level
    'ugly duckling' deviations are z-scores of a lesion against the patient's
    own other lesions; these are computed within-patient so they do not leak
    across the fold boundary, but normalization constants are still fit on train.

    Categorical TARGET ENCODING is the one place a label is touched: it is fit
    here, on the train fold only, with a smoothed mean. The fitted mapping and
    the train global prior go into the state dict so `transform_fold_features`
    can apply them by pure lookup (no recomputation, no leakage).

    State dict keys:
      col_means / col_stds : {col -> float} for global z-scores (size/colour/shape)
      te_prior             : float, train global target mean (fallback for unseen)
      te_maps              : {cat_col -> {category_value -> smoothed_target_mean}}
    """
    state: dict = {"col_means": {}, "col_stds": {}, "te_prior": 0.0, "te_maps": {}}

    for c in SIZE_COLS + COLOR_COLS + SHAPE_COLS:
        if c in train.columns:
            state["col_means"][c] = float(train[c].mean())
            state["col_stds"][c] = float(train[c].std() + 1e-6)

    # --- per-fold smoothed target encoding (TRAIN ONLY) ------------------
    if TARGET in train.columns:
        prior = float(train[TARGET].mean())
        state["te_prior"] = prior
        for c in CAT_COLS:
            if c not in train.columns:
                continue
            grp = train.groupby(c, observed=True)[TARGET]
            stats = grp.agg(["mean", "count"])
            # smoothed = (count*mean + smoothing*prior) / (count + smoothing)
            smoothed = (
                (stats["count"] * stats["mean"] + TE_SMOOTHING * prior)
                / (stats["count"] + TE_SMOOTHING)
            )
            # cast keys to str so test-time categories (read as str) line up,
            # and so NaN/float category labels don't poison the lookup
            state["te_maps"][c] = {str(k): float(v) for k, v in smoothed.items()}

    return state


def transform_fold_features(df: pd.DataFrame, state: dict) -> pd.DataFrame:
    """Apply fitted stats + within-patient deviations. Used on train and val alike.

    Strictly read-only w.r.t. the target: target encodings are applied by lookup
    into `state["te_maps"]`; unseen categories fall back to the train global prior
    in `state["te_prior"]`. No statistic here is recomputed from the input frame's
    labels, so this is safe to call on val and on the held-out test set.
    """
    out = base_features(df)

    # global z-scores using TRAIN-fitted constants
    for c, m in state["col_means"].items():
        s = state["col_stds"][c]
        out[f"z_{c}"] = (out[c] - m) / s

    # within-patient "ugly-duckling" deviation: how unusual is this lesion vs the
    # patient's OWN other lesions. Computed within-patient on whatever split it is
    # applied to (train / val / test each in isolation), so it never reads across
    # the fold boundary and uses NO target.
    if WIDE_DEV:
        # Step-2: generalized to a wide numeric set (every raw tbp_lv_* numeric
        # column + the key engineered size/colour ratios), z-score + min-max rank.
        out = _patient_deviations(out)
    elif GROUP in out.columns:
        # Step-1 baseline: narrow z-score deviation over size + colour bases only.
        for c in SIZE_COLS + COLOR_COLS:
            if c in out.columns:
                g = out.groupby(GROUP)[c]
                out[f"pdev_{c}"] = (out[c] - g.transform("mean")) / (g.transform("std") + 1e-6)

    # Step-3a: within-patient deviation SLICED BY body site / location. Same
    # leak-safety as the plain within-patient deviations (target-free, groups
    # keyed on patient_id so they never cross the fold boundary).
    out = _patient_category_deviations(out)

    # patient-count + within-patient size rank (target-free)
    out = _patient_features(out)

    # --- target encodings: pure lookup of TRAIN-fitted smoothed means ----
    prior = float(state.get("te_prior", 0.0))
    for c, mapping in state.get("te_maps", {}).items():
        if c in out.columns:
            out[f"te_{c}"] = out[c].astype(str).map(mapping).astype(float).fillna(prior)

    return out


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All model-input columns: engineered + numeric, excluding ids/target/leaks.

    Two exclusions matter:
      * non-numeric columns — raw categoricals (read as the pandas-3.0 ``str``
        dtype, NOT ``object``) never reach LightGBM; their signal enters via the
        numeric ``te_*`` target encodings. Filter on ``is_numeric_dtype``, not
        ``!= object``, so the pandas-3.0 string dtype is handled correctly.
      * LEAK_COLS — train-only / post-hoc-diagnosis fields, several of which are
        float64 and would otherwise slip through the numeric filter.
    """
    import pandas.api.types as pdt

    drop = {ID, GROUP, TARGET, "fold", "patient_id"} | LEAK_COLS
    return [c for c in df.columns if c not in drop and pdt.is_numeric_dtype(df[c])]
