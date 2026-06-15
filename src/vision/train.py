"""Per-fold training of the image expert -> leak-free OOF prob + embeddings.

Reads the FROZEN folds; trains one model per fold on that fold's train side;
predicts the held-out fold. AMP on for the 16GB GPU. Outputs an OOF parquet
(prob + embedding columns) consumed by the combiner, plus the cost row from
`efficiency.measure` so the model becomes one point on the frontier.

Config-driven knobs (all optional; defaults follow the proven ISIC-2024 recipe):
  - loss      : {"name": bce|focal|weighted_bce|pauc_surrogate, ...} (see loss.make_loss)
  - sampler   : {"name": neg_subsample|oversample|none, ...}
                neg_subsample (DEFAULT): per-epoch negative subsampling, keys
                  ``neg_ratio`` (default 7), ``pos_mult`` (default 2). Each epoch
                  = all positives (xpos_mult) + fresh n_pos*neg_ratio negatives.
                oversample: the old WeightedRandomSampler, key ``pos_frac``.
                none: plain shuffle over the full fold (slow; debugging only).
  - aug       : {"variant": transV2|light|minimal}                 train augmentation
  - tta       : {"n_tta": 1}                                       inference TTA (1 == off)
  - sched     : {"name": cosine|none, "warmup_frac": 0.05}         LR schedule
  - lr        : AdamW lr (default 1e-4; use 1e-5 for ViT/eva02-heavy)
  - weight_decay : AdamW weight decay (default 1e-3)
  - ema       : {"enabled": false, "decay": 0.999}                 weight EMA
                When enabled, an exponential-moving-average shadow of the weights
                is kept and used for ALL evaluation/prediction (per-epoch val pAUC
                and the final OOF). Training itself uses the raw weights -- EMA is
                eval-only, so it never touches the loss/metric (leak-safe). 2nd
                place used decay 0.995; default here 0.999.
  - mixup     : {"enabled": false, "alpha": 0.2}                   input mixup
                Convex mixup of inputs AND (soft) targets within the training
                batch (Zhang et al., 2018). Applied to TRAIN batches only; the
                BCE+label-smoothing loss already accepts soft float targets, so
                the mixed target flows straight through. 2nd/4th place used it.
                Val/OOF are never mixed -- the metric stays honest.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler

from .. import cv
from ..data import ID, TARGET
from ..efficiency import measure
from . import backbones
from . import loss as L
from .dataset import (
    LesionDataset,
    ResampledLesionDataset,
    apply_tta,
    eval_aug,
    sample_weights,
    train_aug,
    tta_transforms,
)


def _sampler_kind(sampler_cfg: Optional[dict]) -> str:
    """Normalize the sampler name. Default is per-epoch negative subsampling."""
    get = (sampler_cfg.get if hasattr(sampler_cfg, "get") else (lambda k, d=None: d))
    name = str(get("name", "neg_subsample")).lower() if sampler_cfg else "neg_subsample"
    if name in ("neg_subsample", "subsample", "undersample", "per_epoch"):
        return "neg_subsample"
    if name in ("oversample", "weighted", "balanced"):
        return "oversample"
    if name in ("none", "", "off", "false"):
        return "none"
    raise ValueError(
        f"unknown sampler {name!r}; choose neg_subsample | oversample | none"
    )


def _build_weighted_sampler(tr: pd.DataFrame, sampler_cfg: Optional[dict]):
    """The old WeightedRandomSampler path (oversample), kept config-selectable."""
    get = (sampler_cfg.get if hasattr(sampler_cfg, "get") else (lambda k, d=None: d))
    pos_frac = float(get("pos_frac", 0.5))
    w = sample_weights(tr, pos_frac=pos_frac)
    return WeightedRandomSampler(
        weights=torch.as_tensor(w, dtype=torch.double),
        num_samples=len(tr),
        replacement=True,
    )


def _build_scheduler(opt, sched_cfg: Optional[dict], total_steps: int):
    """Cosine schedule with short linear warmup (per-step). ``none`` -> no sched."""
    get = (sched_cfg.get if hasattr(sched_cfg, "get") else (lambda k, d=None: d))
    name = str(get("name", "cosine")).lower() if sched_cfg is not None else "cosine"
    if name in ("none", "", "off", "false") or total_steps <= 0:
        return None
    warmup_frac = float(get("warmup_frac", 0.05))
    warmup = max(1, int(round(total_steps * warmup_frac)))
    min_lr_frac = float(get("min_lr_frac", 0.01))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        prog = (step - warmup) / max(1, total_steps - warmup)
        prog = min(1.0, max(0.0, prog))
        cos = 0.5 * (1.0 + math.cos(math.pi * prog))
        return min_lr_frac + (1.0 - min_lr_frac) * cos

    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)


class EMA:
    """Exponential moving average of model weights (eval-only shadow).

    Keeps a CPU-free, on-device shadow copy of every floating-point parameter
    AND buffer (so BatchNorm running stats are tracked too). ``update`` is called
    after every optimizer step; ``apply_to``/``restore`` swap the shadow weights
    into the live model around evaluation so prediction uses the EMA weights while
    training keeps using the raw ones. This is strictly an eval-time substitution
    -- the EMA never participates in the loss, so it is leak-safe.

    decay close to 1.0 (0.999 default; 2nd place used 0.995) -> slowly tracking,
    well-regularized weights that typically val better than the raw end-of-epoch
    weights on this tail metric.

    IMPORTANT (tiny-epoch regime): per-epoch negative undersampling makes each
    epoch only ~tens of optimizer steps, so a fixed decay like 0.999 has a time
    constant (~1/(1-decay) = 1000 steps) far longer than the whole run -> the
    shadow stays stuck near the random init and val *worse* than the raw weights
    (measured: 0.999 gave fold-0 pAUC 0.079 vs 0.156 raw). We fix this with
    timm-style EMA warmup: the EFFECTIVE decay ramps in as
    ``min(decay, (1+step)/(10+step))`` so EMA tracks fast for the first steps and
    only settles toward the configured decay once enough steps have accrued. This
    keeps the proven EMA behavior on long schedules while making it safe (and
    helpful) on the short undersampled ones.
    """

    def __init__(self, model: torch.nn.Module, decay: float = 0.999,
                 warmup: bool = True):
        self.decay = float(decay)
        self.warmup = bool(warmup)
        self.step = 0
        # shadow holds a detached clone of params + buffers (float tensors only).
        self.shadow = {
            k: v.detach().clone()
            for k, v in model.state_dict().items()
            if v.dtype.is_floating_point
        }
        self._backup: dict = {}

    def _effective_decay(self) -> float:
        if not self.warmup:
            return self.decay
        # timm ModelEmaV2 warmup ramp: small at first, -> self.decay as step grows.
        return min(self.decay, (1.0 + self.step) / (10.0 + self.step))

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.step += 1
        d = self._effective_decay()
        msd = model.state_dict()
        for k, s in self.shadow.items():
            v = msd[k]
            # buffers like num_batches_tracked are int -> skipped (not in shadow).
            s.mul_(d).add_(v.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module) -> None:
        """Swap EMA weights into the model, stashing the raw weights for restore."""
        msd = model.state_dict()
        self._backup = {k: msd[k].detach().clone() for k in self.shadow}
        for k, s in self.shadow.items():
            msd[k].copy_(s)

    @torch.no_grad()
    def restore(self, model: torch.nn.Module) -> None:
        """Put the raw (training) weights back after an EMA evaluation."""
        if not self._backup:
            return
        msd = model.state_dict()
        for k, b in self._backup.items():
            msd[k].copy_(b)
        self._backup = {}


def _mixup_batch(x: torch.Tensor, y: torch.Tensor, alpha: float, gen):
    """Convex mixup of a batch with itself under a Beta(alpha, alpha) lambda.

    Returns (x_mixed, y_mixed) where both inputs and (soft) targets are blended
    by the SAME lam, using a random permutation of the batch as the partner. The
    downstream BCE+label-smoothing loss accepts soft float targets directly, so
    the mixed target needs no special handling. TRAIN-side only.
    """
    if alpha <= 0.0:
        return x, y
    lam = float(np.random.default_rng(int(gen.integers(0, 2**31 - 1))).beta(alpha, alpha))
    # symmetric: keep lam>=0.5 so the dominant image label dominates (stabler).
    lam = max(lam, 1.0 - lam)
    perm = torch.randperm(x.size(0), device=x.device)
    x_mix = lam * x + (1.0 - lam) * x[perm]
    y_mix = lam * y + (1.0 - lam) * y[perm]
    return x_mix, y_mix


@torch.no_grad()
def _predict_fold(model, dl_va, device: str, n_tta: int = 1):
    """OOF prob + embedding for one held-out fold, with optional geometric TTA.

    n_tta == 1 -> deterministic single forward (identity view only). n_tta > 1
    averages sigmoid probs over flip/rot90 views; the embedding is taken from
    the identity view so the stacked features stay well-defined.
    """
    ops = tta_transforms(n_tta)
    probs, embs, idx = [], [], []
    model.eval()
    with torch.amp.autocast("cuda"):
        for x, _, ids in dl_va:
            x = x.to(device)
            # identity view first: gives both logit and embedding
            logit0, z = model(apply_tta(x, ops[0]), return_embedding=True)
            p = torch.sigmoid(logit0).float()
            for op in ops[1:]:
                p = p + torch.sigmoid(model(apply_tta(x, op))).float()
            p = p / float(len(ops))
            probs.append(p.cpu().numpy())
            embs.append(z.float().cpu().numpy())
            idx += list(ids)
    return np.concatenate(probs), np.concatenate(embs), idx


def _eval_state_dict(model: torch.nn.Module,
                     ema_obj: Optional["EMA"]) -> dict:
    """The state_dict we EVALUATE/PREDICT with -> what we ship per fold.

    Without EMA this is just the trained model's weights. With EMA we return a
    FULL state_dict (every key the model has) whose floating-point entries are
    replaced by the EMA shadow values, while integer buffers (e.g. BatchNorm's
    ``num_batches_tracked``, which EMA never tracks) are kept from the live
    model. The result is therefore complete and loads with no missing/unexpected
    keys into a fresh ImageExpert, and exactly mirrors ``EMA.apply_to`` -- i.e.
    the weights used to produce the reported OOF. Everything is detached and
    moved to CPU so the saved file is device-independent.
    """
    msd = model.state_dict()
    if ema_obj is None:
        return {k: v.detach().cpu().clone() for k, v in msd.items()}
    out = {}
    for k, v in msd.items():
        src = ema_obj.shadow[k] if k in ema_obj.shadow else v
        out[k] = src.detach().cpu().clone()
    return out


def _save_fold_ckpt(model: torch.nn.Module, ema_obj: Optional["EMA"],
                    out_dir: str, name: str, fold: int) -> Path:
    """Save the per-fold EVAL weights to ``{out_dir}/ckpt/{name}_fold{k}.pt``.

    A plain ``torch.save`` of the state_dict (EMA shadow when EMA is on). This is
    additive: it does not touch the OOF / embeddings / reported pAUC.
    """
    ckpt_dir = Path(out_dir) / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{name}_fold{int(fold)}.pt"
    torch.save(_eval_state_dict(model, ema_obj), path)
    return path


def _write_manifest(out_dir: str, name: str, backbone: str, timm_id: str,
                    img_size: int, embed_dim: int, folds: list, ema: bool) -> Path:
    """Write the rebuild recipe alongside the per-fold checkpoints.

    Captures everything kaggle_stack_inference.py needs to reconstruct the exact
    ImageExpert and locate its fold checkpoints, without importing the training
    config.
    """
    ckpt_dir = Path(out_dir) / "ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"{name}_manifest.json"
    manifest = {
        "name": name,
        "backbone": backbone,
        "timm_id": timm_id,
        "img_size": int(img_size),
        "embed_dim": int(embed_dim),
        "folds": [int(k) for k in folds],
        "ema": bool(ema),
    }
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def run(meta: pd.DataFrame, folds: pd.DataFrame, hdf5_path: str, backbone: str,
        img_size: int = 128, epochs: int = 8, bs: int = 256, lr: float = 1e-4,
        weight_decay: float = 1e-3, loss: Optional[dict] = None,
        sampler: Optional[dict] = None, tta: Optional[dict] = None,
        aug: Optional[dict] = None, sched: Optional[dict] = None,
        ema: Optional[dict] = None, mixup: Optional[dict] = None,
        num_workers: int = 8, seed: int = 42, only_folds: Optional[list] = None,
        verbose: bool = True, device: str = "cuda",
        name: Optional[str] = None, eval_every: int = 1,
        save_ckpt: bool = True, ckpt_dir: str = "experiments") -> dict:
    df = meta.merge(folds[[ID, "fold"]], on=ID, how="left").reset_index(drop=True)
    assert df["fold"].notna().all(), "run src.cv first to freeze folds"
    y = df[TARGET].values
    oof = np.zeros(len(df))
    embeds = None

    tta_get = (tta.get if hasattr(tta, "get") else (lambda k, d=None: d))
    n_tta = int(tta_get("n_tta", 1)) if tta else 1
    aug_get = (aug.get if hasattr(aug, "get") else (lambda k, d=None: d))
    aug_variant = str(aug_get("variant", "transV2")) if aug else "transV2"
    samp_get = (sampler.get if hasattr(sampler, "get") else (lambda k, d=None: d))
    kind = _sampler_kind(sampler)

    ema_get = (ema.get if hasattr(ema, "get") else (lambda k, d=None: d))
    ema_on = bool(ema_get("enabled", False)) if ema else False
    ema_decay = float(ema_get("decay", 0.999)) if ema else 0.999
    ema_warmup = bool(ema_get("warmup", True)) if ema else True
    mix_get = (mixup.get if hasattr(mixup, "get") else (lambda k, d=None: d))
    mix_on = bool(mix_get("enabled", False)) if mixup else False
    mix_alpha = float(mix_get("alpha", 0.2)) if mixup else 0.0
    if not mix_on:
        mix_alpha = 0.0

    fold_paucs: dict = {}
    saved_folds: list = []
    saved_embed_dim: int = 0
    all_folds = sorted(df["fold"].unique())
    folds_to_run = (all_folds if only_folds is None
                    else [k for k in all_folds if k in set(only_folds)])

    for k in folds_to_run:
        tr = df[df["fold"] != k]
        va = df[df["fold"] == k]
        torch.manual_seed(seed)
        model = backbones.build(backbone, img_size=img_size).to(device)

        n_pos = int((tr[TARGET].values == 1).sum())
        n_neg = int((tr[TARGET].values == 0).sum())
        crit = L.make_loss(loss, n_pos=n_pos, n_neg=n_neg).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = torch.amp.GradScaler("cuda")
        ema_obj = EMA(model, decay=ema_decay, warmup=ema_warmup) if ema_on else None
        mix_rng = np.random.default_rng(seed + 1000 + int(k))

        tf_tr = train_aug(img_size, variant=aug_variant)

        # --- build the TRAIN loader per the sampler kind ---
        resampled_ds = None
        if kind == "neg_subsample":
            resampled_ds = ResampledLesionDataset(
                tr, hdf5_path, tf_tr,
                neg_ratio=float(samp_get("neg_ratio", 7.0)),
                pos_mult=int(samp_get("pos_mult", 2)),
                seed=seed + int(k),
            )
            dl_tr = DataLoader(resampled_ds, batch_size=bs, shuffle=True,
                               num_workers=num_workers, pin_memory=True, drop_last=False)
            steps_per_epoch = max(1, math.ceil(len(resampled_ds) / bs))
        else:
            sampler_obj = _build_weighted_sampler(tr, sampler) if kind == "oversample" else None
            dl_tr = DataLoader(
                LesionDataset(tr, hdf5_path, tf_tr), batch_size=bs,
                shuffle=(sampler_obj is None), sampler=sampler_obj,
                num_workers=num_workers, pin_memory=True, drop_last=True,
            )
            steps_per_epoch = max(1, len(tr) // bs)

        dl_va = DataLoader(LesionDataset(va, hdf5_path, eval_aug(img_size)),
                           batch_size=bs, shuffle=False,
                           num_workers=num_workers, pin_memory=True)

        scheduler = _build_scheduler(opt, sched, total_steps=epochs * steps_per_epoch)

        if verbose:
            ep_imgs = len(resampled_ds) if resampled_ds is not None else len(tr)
            reg = (f"ema={ema_decay if ema_on else 'off'} "
                   f"mixup={mix_alpha if mix_on else 'off'}")
            print(f"[fold {k}] backbone={backbone} img={img_size} sampler={kind} "
                  f"epoch_imgs~{ep_imgs} steps/ep={steps_per_epoch} {reg} "
                  f"(train n_pos={n_pos} n_neg={n_neg}, val rows={len(va)})", flush=True)

        for ep in range(epochs):
            if resampled_ds is not None:
                resampled_ds.resample()   # fresh negatives + reshuffle each epoch
            model.train()
            t0 = time.perf_counter()
            run_loss, nb = 0.0, 0
            for x, yb, _ in dl_tr:
                x, yb = x.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                if mix_on and mix_alpha > 0.0:
                    x, yb = _mixup_batch(x, yb, mix_alpha, mix_rng)
                opt.zero_grad()
                with torch.amp.autocast("cuda"):
                    out = crit(model(x), yb)
                scaler.scale(out).backward()
                scaler.step(opt)
                scaler.update()
                if scheduler is not None:
                    scheduler.step()
                if ema_obj is not None:
                    ema_obj.update(model)
                run_loss += float(out.detach())
                nb += 1
            dt = time.perf_counter() - t0
            # The per-epoch diagnostic eval is over the FULL ~80k-row fold and at
            # 224-336px dominates wall-clock; ``eval_every`` throttles it (the
            # final OOF is always computed once after all epochs regardless). The
            # last epoch is always evaluated so the printed trajectory ends on the
            # shipped weights. eval_every=1 (default) preserves the old behavior.
            do_eval = verbose and ((ep + 1) % max(1, eval_every) == 0
                                   or (ep + 1) == epochs)
            if do_eval:
                # honest per-epoch val pAUC on the FULL held-out fold (EMA weights
                # if EMA is on -- mirrors what the final OOF will use).
                if ema_obj is not None:
                    ema_obj.apply_to(model)
                p_ep, _, ids_ep = _predict_fold(model, dl_va, device, n_tta=1)
                if ema_obj is not None:
                    ema_obj.restore(model)
                pauc_ep = cv.oof_pauc(va[TARGET].values, _align(p_ep, ids_ep, va))
                lr_now = opt.param_groups[0]["lr"]
                print(f"[fold {k}] ep {ep+1:02d}/{epochs} loss={run_loss/max(nb,1):.4f} "
                      f"val_pAUC@80={pauc_ep:.5f} lr={lr_now:.2e} {dt:.1f}s/epoch",
                      flush=True)

        # Final OOF prediction uses the EMA weights when EMA is on (the EMA
        # shadow is the model we "ship" for this fold). Leak-safe: still the FULL
        # held-out fold, never the mixed/subsampled train side.
        if ema_obj is not None:
            ema_obj.apply_to(model)
        p, e, ids_va = _predict_fold(model, dl_va, device, n_tta=n_tta)
        if ema_obj is not None:
            ema_obj.restore(model)
        pos = df.index[df["fold"] == k].to_numpy()
        oof[pos] = _align(p, ids_va, va)
        if embeds is None:
            embeds = np.zeros((len(df), e.shape[1]), dtype=np.float32)
        embeds[pos] = _align_2d(e, ids_va, va)
        fold_paucs[int(k)] = float(cv.oof_pauc(va[TARGET].values, _align(p, ids_va, va)))

        # --- per-fold checkpoint (additive; never touches the OOF above) ---
        # We save the EVAL weights -- the EMA shadow when EMA is on -- read
        # straight from ``_eval_state_dict`` (independent of whether EMA is
        # currently swapped into the live model). One file per fold.
        if save_ckpt:
            out_name = name or backbone
            ck = _save_fold_ckpt(model, ema_obj, ckpt_dir, out_name, int(k))
            if verbose:
                print(f"[fold {k}] saved checkpoint -> {ck} "
                      f"(eval weights{' = EMA shadow' if ema_obj is not None else ''})",
                      flush=True)
            saved_folds.append(int(k))
            saved_embed_dim = int(model.embed_dim)

    # --- manifest: the rebuild recipe for kaggle_stack_inference (additive) ---
    # Written once all requested folds are trained+saved. Note: when only a
    # SUBSET of folds is run (smoke), ``folds`` lists just those -- rerunning the
    # full sweep overwrites it with the complete [0..4] set.
    if save_ckpt and saved_folds:
        out_name = name or backbone
        timm_id = backbones.FRONTIER.get(backbone, backbone)
        mpath = _write_manifest(
            ckpt_dir, out_name, backbone, timm_id, img_size,
            saved_embed_dim, saved_folds, ema=ema_on,
        )
        if verbose:
            print(f"wrote manifest -> {mpath}  folds={saved_folds}", flush=True)

    # If only a subset of folds ran, score only those rows (single-fold smoke).
    if only_folds is not None:
        mask = df["fold"].isin(set(folds_to_run)).to_numpy()
        score = cv.oof_pauc(y[mask], oof[mask])
    else:
        score = cv.oof_pauc(y, oof)
    cost = measure(backbones.build(backbone, pretrained=False, img_size=img_size), img_size)
    # ``name`` is the OOF/column tag (lets convnextv2_nano@224 write a distinct
    # parquet from the @128 one); ``backbone`` stays the timm arch for the cost row.
    out_name = name or backbone
    return {"backbone": backbone, "name": out_name, "oof": oof, "embeds": embeds,
            "pauc": score, "fold_paucs": fold_paucs, "cost": cost.as_row(),
            ID: df[ID].values}


def _align(p: np.ndarray, ids: list, va: pd.DataFrame) -> np.ndarray:
    """Reorder a per-id prediction vector to match ``va`` row order.

    _predict_fold returns predictions in DataLoader iteration order; map them
    back to the fold's row order by isic_id so they line up with df.index slots.
    """
    order = {str(i): j for j, i in enumerate(ids)}
    sel = [order[str(i)] for i in va[ID].values]
    return p[sel]


def _align_2d(e: np.ndarray, ids: list, va: pd.DataFrame) -> np.ndarray:
    order = {str(i): j for j, i in enumerate(ids)}
    sel = [order[str(i)] for i in va[ID].values]
    return e[sel]


def save(result: dict, out_dir: str = "experiments") -> None:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    name = result.get("name", result["backbone"])
    cols = {ID: result[ID], f"{name}_oof": result["oof"]}
    E = result["embeds"]
    for j in range(E.shape[1]):
        cols[f"{name}_emb{j}"] = E[:, j]
    pd.DataFrame(cols).to_parquet(f"{out_dir}/vision_{name}_oof.parquet", index=False)
