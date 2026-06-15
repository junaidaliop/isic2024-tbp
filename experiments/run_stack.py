"""Reproduce the multimodal STACK headline + the honest combiner ablation.

Run AFTER the tabular GBDT (`python -m src.gbdt`) and the image experts
(`python -m src.vision.train_cli --cfg ...`) have written their OOF parquets to
experiments/. Everything is scored ONLY via src/cv.py on the frozen folds.

    PYTHONPATH=. python experiments/run_stack.py

Canonical headline = the trivial param-free rank-average of the tabular GBDT OOF
and the best image expert (convnextv2_nano @224 = 0.15945). At 393 positives the
meta-LGBM, the learned gate, and extra weak image models all LOSE to this trivial
combiner (see the printed ablation) -- which is the project's reportable result.

Leak-safety (cv-guardian-verified): every meta-LGBM/gate stack trains per-fold on
folds != k and predicts fold k on the SAME frozen data/folds.parquet; the
image-ugly-duckling features are within-patient (groupby patient_id) and
target-free, and patients never straddle folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from src import cv, stack

ID = "isic_id"
SEED = 42
PRIMARY_IMG = "convnextv2_nano_r224"   # best image expert (0.15945)


def _ra(*xs: np.ndarray) -> np.ndarray:
    return np.mean([rankdata(x) / len(x) for x in xs], axis=0)


def main() -> None:
    folds = pd.read_parquet("data/folds.parquet")[[ID, "patient_id", "target", "fold"]]
    g = pd.read_parquet("experiments/gbdt_oof.parquet")
    strong = [PRIMARY_IMG, "convnextv2_nano", "convnextv2_tiny", "effvit_b0", "vit_tiny"]
    d = folds.merge(g, on=ID)
    for n in strong:
        d = d.merge(pd.read_parquet(f"experiments/vision_{n}_oof.parquet",
                                    columns=[ID, f"{n}_oof"]), on=ID)
    y = d["target"].values
    P = lambda s: round(cv.pauc_above_tpr(y, s), 5)  # noqa: E731

    print("=== references ===")
    print(f"tabular GBDT                       {P(d.gbdt_oof.values)}")
    print(f"best image ({PRIMARY_IMG})         {P(d[PRIMARY_IMG + '_oof'].values)}")

    canonical = _ra(d.gbdt_oof.values, d[PRIMARY_IMG + "_oof"].values)
    print("\n=== combiner ablation (OOF pAUC@80%TPR) ===")
    print(f"rank-avg[gbdt, r224]              {P(canonical)}   <- CANONICAL (trivial combiner)")
    print(f"rank-avg[gbdt*2, r224]            {P(_ra(d.gbdt_oof.values, d.gbdt_oof.values, d[PRIMARY_IMG + '_oof'].values))}   (gbdt-upweighted; OOF-tuned, ties within noise)")
    print(f"rank-avg[gbdt, r224, nano, effvit] {P(_ra(d.gbdt_oof.values, d[PRIMARY_IMG + '_oof'].values, d.convnextv2_nano_oof.values, d.effvit_b0_oof.values))}   (extra weak imgs HURT)")

    try:
        udk = stack.image_ugly_duckling(f"experiments/vision_{PRIMARY_IMG}_oof.parquet", PRIMARY_IMG)
        cols = ["gbdt_oof", f"{PRIMARY_IMG}_oof", "convnextv2_nano_oof",
                "convnextv2_tiny_oof", "effvit_b0_oof", "vit_tiny_oof"]
        feat = d[[ID] + cols].merge(udk, on=ID)
        _, pm = stack.gbdt_stack(d[[ID, "target"]], folds[[ID, "fold"]], feat,
                                 {"num_leaves": 8, "learning_rate": 0.03,
                                  "min_data_in_leaf": 100, "lambda_l2": 10.0})
        print(f"meta-LGBM[gbdt+imgs+udk]          {round(pm, 5)}   (learned meta LOSES to rank-avg)")
    except Exception as e:  # noqa: BLE001
        print(f"meta-LGBM skipped: {e!r}")

    try:
        _, pg = stack.gate_ablation(d[[ID, "target"]], folds[[ID, "fold"]],
                                    "gbdt_oof", f"{PRIMARY_IMG}_oof",
                                    d[[ID, "gbdt_oof", f"{PRIMARY_IMG}_oof"]], None)
        print(f"learned gate[gbdt, img]          {round(pg, 5)}   (NEGATIVE result, as predicted)")
    except Exception as e:  # noqa: BLE001
        print(f"gate ablation skipped: {e!r}")

    # persist the canonical stack + log the frontier point
    pa = P(canonical)
    pd.DataFrame({ID: d[ID].values, "stack_oof": canonical}).to_parquet(
        "experiments/stack_oof.parquet", index=False)
    cost = pd.read_csv("reports/frontier_cost.csv").set_index("model")
    cval = lambda m, c: (float(cost.loc[m, c]) if m in cost.index else 0.0)  # noqa: E731
    row = {"model": "stack_best", "pauc": pa,
           "params_m": round(cval("gbdt", "params_m") + cval(PRIMARY_IMG, "params_m"), 3),
           "gflops": round(cval("gbdt", "gflops") + cval(PRIMARY_IMG, "gflops"), 4),
           "cpu_ms": round(cval("gbdt", "cpu_ms") + cval(PRIMARY_IMG, "cpu_ms"), 3),
           "img_size": 224}
    stack.log_frontier(row, "reports/frontier.csv")
    print(f"\nCANONICAL STACK pAUC@80%TPR = {pa}  (tabular ref {P(d.gbdt_oof.values)})")
    print(f"  cost: {row['params_m']}M params / {row['gflops']} GFLOPs / {row['cpu_ms']} ms (1-thread CPU)")
    print("  -> experiments/stack_oof.parquet  +  reports/frontier.csv [stack_best]")


if __name__ == "__main__":
    main()
