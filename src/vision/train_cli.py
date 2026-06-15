"""CLI: train one image expert from a config and log its frontier point.

    python -m src.vision.train_cli --cfg configs/vision/mnv4_small.yaml
"""
from __future__ import annotations

import argparse

import pandas as pd

from .. import config as C
from .. import cv, stack
from . import train as T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--base", default="configs/default.yaml")
    ap.add_argument("--folds", default=None,
                    help="comma-separated fold ids to run (e.g. '0' for a smoke test); "
                         "default = all 5 folds")
    ap.add_argument("--epochs", type=int, default=None, help="override config epochs")
    ap.add_argument("--no-log", action="store_true",
                    help="skip writing the frontier CSV (use for smoke runs)")
    a = ap.parse_args()

    cfg = C.load(a.base, a.cfg)
    meta = pd.read_csv(cfg.paths.train_meta, low_memory=False)
    folds = cv.load_folds(cfg.paths.folds)

    only_folds = None
    if a.folds is not None:
        only_folds = [int(x) for x in str(a.folds).split(",") if x.strip() != ""]

    v = cfg.vision
    res = T.run(meta, folds, cfg.paths.train_hdf5, v.backbone,
                img_size=v.img_size, epochs=(a.epochs or v.epochs),
                bs=v.batch_size, lr=v.lr,
                weight_decay=v.get("weight_decay", 1e-3),
                loss=v.get("loss"), sampler=v.get("sampler"),
                tta=v.get("tta"), aug=v.get("aug"), sched=v.get("sched"),
                ema=v.get("ema"), mixup=v.get("mixup"),
                num_workers=int(v.get("num_workers", 8)),
                seed=int(cfg.get("seed", 42)),
                only_folds=only_folds,
                name=v.get("name"),
                eval_every=int(v.get("eval_every", 1)))
    T.save(res, cfg.paths.experiments)

    if not a.no_log:
        row = {"model": res.get("name", res["backbone"]),
               "pauc": round(res["pauc"], 5), **res["cost"]}
        stack.log_frontier(row, f"{cfg.paths.reports}/frontier.csv")
    print(f"{res.get('name', res['backbone'])}: pAUC={res['pauc']:.5f}  "
          f"fold_paucs={res['fold_paucs']}  cost={res['cost']}")


if __name__ == "__main__":
    main()
