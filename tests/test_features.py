"""Leak-safety + shape tests for tabular features. Run: pytest -q"""
import numpy as np
import pandas as pd

from src import features as F


def _toy(n=2000, n_pat=50, seed=0):
    rng = np.random.default_rng(seed)
    pid = rng.integers(0, n_pat, n)
    sites = np.array(["torso", "upper extremity", "lower extremity", "head/neck"])
    locs = np.array(["Torso Back", "Left Arm", "Right Leg", "Face", "Torso Front"])
    df = pd.DataFrame({
        "isic_id": [f"ISIC_{i:06d}" for i in range(n)],
        "patient_id": [f"P{p:03d}" for p in pid],
        "target": (rng.random(n) < 0.01).astype(int),
        "clin_size_long_diam_mm": rng.gamma(2, 2, n),
        "tbp_lv_areaMM2": rng.gamma(3, 3, n),
        "tbp_lv_perimeterMM": rng.gamma(3, 2, n),
        "tbp_lv_minorAxisMM": rng.gamma(2, 1.5, n),
        "tbp_lv_area_perim_ratio": rng.gamma(2, 1, n),
        "tbp_lv_norm_border": rng.random(n),
        "tbp_lv_norm_color": rng.random(n),
        "tbp_lv_deltaLBnorm": rng.normal(0, 1, n),
        "tbp_lv_deltaA": rng.normal(0, 1, n),
        "tbp_lv_deltaB": rng.normal(0, 1, n),
        "tbp_lv_deltaL": rng.normal(0, 1, n),
        "tbp_lv_A": rng.normal(0, 1, n),
        "tbp_lv_B": rng.normal(0, 1, n),
        "tbp_lv_C": rng.normal(0, 1, n),
        "tbp_lv_H": rng.random(n) * 60,
        "tbp_lv_Hext": rng.random(n) * 60,
        "tbp_lv_L": rng.random(n) * 100,
        "tbp_lv_Lext": rng.random(n) * 100,
        "tbp_lv_stdL": rng.gamma(2, 1, n),
        "tbp_lv_nevi_confidence": rng.random(n),
        "tbp_lv_x": rng.normal(0, 10, n),
        "tbp_lv_y": rng.normal(0, 10, n),
        "tbp_lv_z": rng.normal(0, 10, n),
        "tbp_lv_color_std_mean": rng.gamma(2, 1, n),
        "tbp_lv_radial_color_std_max": rng.gamma(2, 1, n),
        "tbp_lv_symm_2axis": rng.random(n),
        "tbp_lv_eccentricity": rng.random(n),
        "age_approx": rng.integers(20, 80, n).astype(float),
        "anatom_site_general": rng.choice(sites, n),
        "tbp_lv_location": rng.choice(locs, n),
        "tbp_lv_location_simple": rng.choice(sites, n),
    })
    return df


def test_fold_state_fit_on_train_only():
    df = _toy()
    tr, va = df.iloc[:1500], df.iloc[1500:]
    state = F.fit_fold_features(tr)            # fit on TRAIN only
    out_va = F.transform_fold_features(va, state)
    # z-scores on val use train constants -> mean not forced to 0 on val
    assert "z_clin_size_long_diam_mm" in out_va.columns
    assert abs(out_va["z_clin_size_long_diam_mm"].mean()) > 1e-6


def test_feature_columns_exclude_ids_and_target():
    df = _toy()
    state = F.fit_fold_features(df)
    out = F.transform_fold_features(df, state)
    cols = F.feature_columns(out)
    for forbidden in ["isic_id", "patient_id", "target"]:
        assert forbidden not in cols


def test_feature_columns_excludes_strings_and_leak_cols():
    """Regression: pandas-3.0 'str' dtype must be excluded, and float64
    train-only leak columns (mel_thick_mm, iddx_*) must be dropped explicitly."""
    df = _toy()
    df["sex"] = "male"                       # pandas-3.0 reads as 'str', not object
    df["anatom_site_general"] = "torso"
    df["mel_thick_mm"] = 1.5                 # numeric train-only leak
    df["iddx_2"] = 0.0                        # numeric train-only leak
    df["tbp_lv_dnn_lesion_confidence"] = 0.3  # numeric train-only
    state = F.fit_fold_features(df)
    out = F.transform_fold_features(df, state)
    cols = F.feature_columns(out)
    for forbidden in ["sex", "anatom_site_general", "mel_thick_mm", "iddx_2",
                      "tbp_lv_dnn_lesion_confidence"]:
        assert forbidden not in cols, f"{forbidden} leaked into feature columns"
    # the numeric target-encoding of a raw categorical SHOULD be present
    assert "te_sex" in cols


def test_patient_deviations_leak_safe_and_finite():
    """Wide within-patient ugly-duckling deviations must be: (a) finite, (b)
    independent across the fold boundary, and (c) free of any target column.

    Independence: computing pdev/prank on a val frame uses ONLY that frame's
    within-patient stats, so the values must not change whether or not the train
    rows are present. We verify by transforming val alone vs val concatenated
    after train and checking the val rows are identical."""
    df = _toy(n=3000, n_pat=40)
    # split by PATIENT (mirrors the frozen folds: no patient straddles the line)
    pats = sorted(df["patient_id"].unique())
    va_pats = set(pats[: len(pats) // 4])
    va = df[df["patient_id"].isin(va_pats)].copy()
    tr = df[~df["patient_id"].isin(va_pats)].copy()
    state = F.fit_fold_features(tr)

    out_va_alone = F.transform_fold_features(va, state)
    dev_cols = [c for c in out_va_alone.columns
                if c.startswith("pdev_") or c.startswith("prank_")]
    assert dev_cols, "no within-patient deviation features were produced"

    # finiteness
    import numpy as np
    block = out_va_alone[dev_cols].to_numpy(dtype=float)
    assert np.isfinite(block).all(), "patient deviations produced inf/nan"

    # no target column survives into the feature matrix
    cols = F.feature_columns(out_va_alone)
    assert "target" not in cols

    # cross-fold independence: with patient-disjoint splits, a val patient's
    # within-patient deviations must be identical whether computed on the val
    # frame alone or inside the full frame (train rows of OTHER patients cannot
    # affect them). This is the leak-safety guarantee the frozen folds rely on.
    out_full = F.transform_fold_features(df, state)
    va_in_full = out_full.loc[va.index, dev_cols].to_numpy(dtype=float)
    assert np.allclose(va_in_full, block, equal_nan=True), (
        "within-patient deviations leaked across the val/train boundary"
    )


def test_pxc_features_leak_safe_and_finite():
    """Step-3a patient x category deviations must be: (a) produced, (b) finite,
    (c) free of any target, and (d) independent across the fold boundary.

    Independence: pxc_ deviations are keyed on patient_id (sliced by site), so
    with patient-disjoint splits a val patient's pxc values must be identical
    whether computed on the val frame alone or inside the full frame. This is the
    leak-safety guarantee the frozen patient-grouped folds rely on."""
    df = _toy(n=3000, n_pat=40)
    pats = sorted(df["patient_id"].unique())
    va_pats = set(pats[: len(pats) // 4])
    va = df[df["patient_id"].isin(va_pats)].copy()
    tr = df[~df["patient_id"].isin(va_pats)].copy()
    state = F.fit_fold_features(tr)

    out_va_alone = F.transform_fold_features(va, state)
    pxc_cols = [c for c in out_va_alone.columns if c.startswith("pxc_")]
    assert pxc_cols, "no patient x category deviation features were produced"
    # we capped to a curated numeric set x 3 categoricals; sanity on count
    assert len(pxc_cols) >= 3, f"expected several pxc_ features, got {len(pxc_cols)}"

    block = out_va_alone[pxc_cols].to_numpy(dtype=float)
    assert np.isfinite(block).all(), "pxc deviations produced inf/nan"

    # no target survives into the feature matrix
    cols = F.feature_columns(out_va_alone)
    assert "target" not in cols
    # the raw categoricals used to slice must NOT enter the model (str dtype)
    for cat in ["anatom_site_general", "tbp_lv_location", "tbp_lv_location_simple"]:
        assert cat not in cols, f"{cat} leaked into feature columns"

    # cross-fold independence: a val patient's pxc deviation is computed only
    # from that patient's own lesions, so adding train rows (other patients)
    # cannot change it.
    out_full = F.transform_fold_features(df, state)
    va_in_full = out_full.loc[va.index, pxc_cols].to_numpy(dtype=float)
    assert np.allclose(va_in_full, block, equal_nan=True), (
        "patient x category deviations leaked across the val/train boundary"
    )
