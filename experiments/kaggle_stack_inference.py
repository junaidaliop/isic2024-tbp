"""Full-STACK late-submission inference for the (closed) ISIC-2024 code competition.

The competition is a code competition with a HIDDEN test set, so the only way to
score the multimodal STACK (not just the tabular GBDT) is to run inference inside
a Kaggle notebook. This script reproduces the CANONICAL stack exactly:

    submission = rank_avg( tabular_GBDT_test_prob , image_5fold_avg_test_prob )

i.e. the equal-weight rank-average of the bagged GBDT and the 5-fold-averaged
image expert -- the same trivial combiner that wins CV in src/stack.py /
experiments/run_stack.py (PRIMARY_IMG = convnextv2_nano_r224, OOF pAUC 0.15945).

------------------------------------------------------------------------------
HOW TO RUN ON KAGGLE
------------------------------------------------------------------------------
1. Build a Kaggle DATASET containing:
     - this repository (so `src/` and `experiments/` import; PYTHONPATH=repo root)
     - experiments/gbdt_boosters.joblib            (the bagged tabular boosters)
     - experiments/ckpt/                            (per-fold image checkpoints +
                                                     {name}_manifest.json)
2. In the notebook, ATTACH the competition data
     (`isic-2024-challenge`, providing test-metadata.csv + test-image.hdf5).
3. Run, pointing the paths at the attached inputs, e.g.:

     import sys; sys.path.insert(0, "/kaggle/input/<your-dataset>/isic2024-tbp")
     !python /kaggle/input/<your-dataset>/isic2024-tbp/experiments/kaggle_stack_inference.py \
         --test-meta /kaggle/input/isic-2024-challenge/test-metadata.csv \
         --test-hdf5 /kaggle/input/isic-2024-challenge/test-image.hdf5 \
         --boosters  /kaggle/input/<your-dataset>/isic2024-tbp/experiments/gbdt_boosters.joblib \
         --ckpt-dir  /kaggle/input/<your-dataset>/isic2024-tbp/experiments/ckpt \
         --name      convnextv2_nano_r224 \
         --out       /kaggle/working/submission.csv

   (Kaggle inputs are read-only -> write --out under /kaggle/working/.)

Notes on parity with the reported number:
  - Image inference uses the EVAL transform (`eval_aug`, Resize+Normalize) and
    NO TTA, matching the OOF generation that produced the frontier point.
  - The per-fold checkpoints are the EVAL weights (EMA shadow when EMA was on),
    so the 5-fold average mirrors how the OOF was scored.
  - GBDT prediction reuses submit.predict_gbdt -> each booster reapplies its own
    leak-free per-fold feature transform, then folds are averaged.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from torch.utils.data import DataLoader

from src import submit
from src.data import ID
from src.vision import backbones
from src.vision.dataset import LesionDataset, eval_aug


def load_manifest(ckpt_dir: str, name: str) -> dict:
    """Read {ckpt_dir}/{name}_manifest.json (the rebuild recipe), if present.

    Returns {} when no manifest exists; callers fall back to CLI args. The
    manifest schema is:
      {name, backbone, timm_id, img_size, embed_dim, folds:[...], ema:bool}.
    """
    p = Path(ckpt_dir) / f"{name}_manifest.json"
    if not p.exists():
        return {}
    return json.loads(p.read_text())


def _fold_ckpt_paths(ckpt_dir: str, name: str, folds: list[int]) -> list[Path]:
    paths = []
    for k in folds:
        p = Path(ckpt_dir) / f"{name}_fold{int(k)}.pt"
        if not p.exists():
            raise FileNotFoundError(f"missing fold checkpoint: {p}")
        paths.append(p)
    return paths


@torch.no_grad()
def _predict_image_one_fold(model: torch.nn.Module, dl: DataLoader,
                            device: str) -> tuple[np.ndarray, list[str]]:
    """Sigmoid prob for every test row from ONE fold's weights (no TTA).

    Returns (probs, ids) in DataLoader iteration order. AMP under CUDA; falls
    back to plain fp32 on CPU.
    """
    model.eval()
    probs, ids_out = [], []
    use_amp = str(device).startswith("cuda")
    autocast = (torch.amp.autocast("cuda") if use_amp
                else torch.amp.autocast("cpu", enabled=False))
    with autocast:
        for x, _, ids in dl:
            x = x.to(device)
            logit = model(x)
            probs.append(torch.sigmoid(logit).float().cpu().numpy())
            ids_out += list(ids)
    return np.concatenate(probs), ids_out


def image_test_probs(test_meta: pd.DataFrame, test_hdf5: str, ckpt_dir: str,
                     name: str, *, backbone: str | None = None,
                     img_size: int | None = None, folds: list[int] | None = None,
                     bs: int = 256, num_workers: int = 8,
                     device: str = "cuda") -> np.ndarray:
    """5-fold-averaged image test probability, aligned to ``test_meta`` rows.

    Rebuilds ImageExpert(name) per fold from the manifest (CLI args override),
    loads {ckpt_dir}/{name}_fold{k}.pt, runs eval_aug inference (NO TTA) over the
    HDF5 test crops, then averages the per-fold sigmoid probs. Manifest values
    fill any unspecified backbone/img_size/folds.
    """
    man = load_manifest(ckpt_dir, name)
    backbone = backbone or man.get("backbone", name)
    img_size = int(img_size if img_size is not None else man.get("img_size", 128))
    folds = folds if folds is not None else man.get("folds", [0, 1, 2, 3, 4])

    # build the test loader ONCE (transform is fold-independent); has_target=False
    # because the hidden test carries no labels.
    ds = LesionDataset(test_meta, test_hdf5, eval_aug(img_size), has_target=False)
    dl = DataLoader(ds, batch_size=bs, shuffle=False,
                    num_workers=num_workers, pin_memory=str(device).startswith("cuda"))

    ck_paths = _fold_ckpt_paths(ckpt_dir, name, folds)
    acc = np.zeros(len(test_meta), dtype=np.float64)
    for k, ck in zip(folds, ck_paths):
        # pretrained=False: weights come entirely from the checkpoint, so this
        # never needs internet on Kaggle.
        model = backbones.build(backbone, pretrained=False, img_size=img_size).to(device)
        state = torch.load(ck, map_location=device)
        missing, unexpected = model.load_state_dict(state, strict=True)
        assert not missing and not unexpected, (
            f"fold {k}: state_dict mismatch missing={missing} unexpected={unexpected}")
        p, ids = _predict_image_one_fold(model, dl, device)
        # align this fold's per-id probs back to test_meta row order
        order = {str(i): j for j, i in enumerate(ids)}
        sel = [order[str(i)] for i in test_meta[ID].values]
        acc += p[sel]
        del model
        if str(device).startswith("cuda"):
            torch.cuda.empty_cache()
        print(f"[image] fold {k}: predicted {len(ids)} rows from {ck.name}", flush=True)
    return acc / float(len(ck_paths))


def rank_avg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Equal-weight rank-average of two score vectors (the canonical combiner)."""
    return 0.5 * (rankdata(a) / len(a) + rankdata(b) / len(b))


def main(argv: list[str] | None = None) -> None:
    import joblib

    ap = argparse.ArgumentParser(description="Full-stack ISIC-2024 Kaggle inference.")
    ap.add_argument("--test-meta",
                    default="/kaggle/input/isic-2024-challenge/test-metadata.csv")
    ap.add_argument("--test-hdf5",
                    default="/kaggle/input/isic-2024-challenge/test-image.hdf5")
    ap.add_argument("--boosters", default="experiments/gbdt_boosters.joblib")
    ap.add_argument("--ckpt-dir", default="experiments/ckpt")
    ap.add_argument("--name", default="convnextv2_nano_r224",
                    help="image-expert tag; matches {name}_fold{k}.pt + manifest")
    ap.add_argument("--backbone", default=None,
                    help="override manifest backbone (timm arch key in FRONTIER)")
    ap.add_argument("--img-size", type=int, default=None,
                    help="override manifest img_size")
    ap.add_argument("--folds", default=None,
                    help="comma-separated fold ids (default = manifest / 0..4)")
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    ap.add_argument("--gbdt-only", action="store_true",
                    help="skip image experts; write GBDT-only submission")
    ap.add_argument("--out", default="submission.csv")
    a = ap.parse_args(argv)

    test_meta = pd.read_csv(a.test_meta, low_memory=False)
    assert ID in test_meta.columns, f"{a.test_meta}: missing {ID!r}"

    # --- tabular: bagged GBDT, each booster reapplies its leak-free transform ---
    boosters = joblib.load(a.boosters)
    tab = submit.predict_gbdt(test_meta, boosters)
    print(f"[gbdt] predicted {len(tab)} rows from {len(boosters)} folds", flush=True)

    if a.gbdt_only:
        submit.write_submission(test_meta, tab, a.out)
        print(f"submission mode: GBDT-only -> {a.out}")
        return

    # --- image: 5-fold-averaged eval-aug (no TTA) probs ---
    folds = ([int(x) for x in str(a.folds).split(",") if x.strip() != ""]
             if a.folds is not None else None)
    img = image_test_probs(
        test_meta, a.test_hdf5, a.ckpt_dir, a.name,
        backbone=a.backbone, img_size=a.img_size, folds=folds,
        bs=a.bs, num_workers=a.num_workers, device=a.device,
    )

    # --- combiner: equal-weight rank-average (canonical) ---
    scores = rank_avg(tab, img)
    submit.write_submission(test_meta, scores, a.out)
    print(f"submission mode: STACK rank-avg[gbdt, {a.name}] -> {a.out}")


if __name__ == "__main__":
    main()
