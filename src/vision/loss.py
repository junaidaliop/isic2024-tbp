"""Imbalance-aware losses for ~0.1% positive prevalence.

Four options, selected by config via ``make_loss(cfg)``:

  - "bce"            : THE PROVEN DEFAULT. Plain BCE-with-logits + optional label
                       smoothing (``label_smoothing``, default 0.05). The top
                       ISIC-2024 open-source teams found that, once the class
                       imbalance is handled by per-epoch negative subsampling (see
                       dataset.ResampledLesionDataset), plain BCE+LS beats focal.
                       Targets are softened y -> y*(1-eps) + eps/2, which both
                       regularizes and calibrates the sigmoid OOF readout. Set
                       ``pos_weight`` to also up-weight the positive class (this is
                       the old "weighted_bce" behavior; null/None -> 1.0, i.e. no
                       up-weight, since the sampler already balances the batch).
  - "focal"          : focal loss (Lin et al., 2017). Imbalance-loss ablation;
                       alpha/gamma are config-driven.
  - "weighted_bce"   : alias of "bce" kept for backward compat; defaults
                       ``pos_weight`` to auto = n_neg / n_pos when omitted (the
                       old behavior) instead of 1.0.
  - "pauc_surrogate" : a self-contained squared-hinge AUC-margin surrogate
                       (LibAUC-style pairwise loss, NO external libauc dep). The
                       ablation that asks whether directly optimizing a ranking
                       margin beats a generic imbalance loss on the tail metric
                       (pAUC@80%TPR). Expected to help only marginally at 393
                       positives; report either way.

All losses take (logits, targets) with logits/targets shaped (B,) or (B, 1)
and return a scalar.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as Fn


class FocalLoss(nn.Module):
    """Binary focal loss on logits. alpha weights the positive class."""

    def __init__(self, alpha: float = 0.9, gamma: float = 2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.reshape(-1)
        targets = targets.reshape(-1).float()
        p = torch.sigmoid(logits)
        ce = Fn.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        pt = torch.where(targets == 1, p, 1 - p)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        return (alpha_t * (1 - pt).pow(self.gamma) * ce).mean()


class BCELoss(nn.Module):
    """BCE-with-logits + label smoothing, with an optional positive-class weight.

    The proven default for this task once per-epoch negative subsampling balances
    the batch. Two knobs:

      - ``label_smoothing`` (eps, default 0.05): targets are softened to
        ``y*(1-eps) + eps/2`` before BCE. This regularizes the head and keeps the
        sigmoid outputs from saturating, which yields a better-calibrated OOF
        probability for stacking into the GBDT.
      - ``pos_weight`` (default 1.0): scales the positive (target==1) loss
        contribution exactly as ``nn.BCEWithLogitsLoss(pos_weight=...)``. With the
        balanced sampler this is usually left at 1.0; the weighted_bce alias wires
        the old auto n_neg/n_pos behavior for the ablation.

    Kept a thin nn.Module so the pos_weight tensor follows ``.to(device)``.
    """

    def __init__(self, pos_weight: float = 1.0, label_smoothing: float = 0.05):
        super().__init__()
        self.register_buffer("pos_weight", torch.tensor(float(pos_weight)))
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.reshape(-1)
        targets = targets.reshape(-1).float()
        if self.label_smoothing > 0.0:
            eps = self.label_smoothing
            targets = targets * (1.0 - eps) + eps / 2.0
        return Fn.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight
        )


# Backward-compatible alias: existing code / pickles referencing WeightedBCE.
WeightedBCE = BCELoss


class PAUCSurrogateLoss(nn.Module):
    """Squared-hinge AUC-margin surrogate (LibAUC-style), self-contained.

    Optimizes a pairwise ranking margin between positive and negative scores,
    a smooth surrogate for AUC / partial-AUC. For every in-batch (pos, neg)
    pair we penalize ``max(0, margin - (s_pos - s_neg))^2``; driving this down
    pushes positive scores above negatives by at least ``margin``, which is the
    quantity the pAUC@80%TPR metric rewards in its tail region.

    This is the squared-hinge form of the standard AUC-margin objective used by
    LibAUC (Yuan et al.), implemented here without any external libauc
    dependency. We operate on sigmoid probabilities so the margin lives on the
    bounded [0, 1] scale and is comparable across batches.

    Notes for the extreme-imbalance regime:
      - With oversampling on, batches carry several positives and the pairwise
        loss is well-conditioned. Without any positive in a batch the pairwise
        term is undefined; we fall back to a tiny BCE anchor so the step is not
        wasted (and so logits stay calibrated for the sigmoid OOF readout).
      - ``surrogate="margin"`` (default) uses sigmoid-prob scores; set
        ``on_logits=True`` to compute the margin on raw logits instead.
    """

    def __init__(self, margin: float = 1.0, on_logits: bool = False,
                 bce_anchor: float = 0.05):
        super().__init__()
        self.margin = float(margin)
        self.on_logits = bool(on_logits)
        self.bce_anchor = float(bce_anchor)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        logits = logits.reshape(-1)
        targets = targets.reshape(-1).float()
        scores = logits if self.on_logits else torch.sigmoid(logits)

        pos_mask = targets == 1
        neg_mask = ~pos_mask
        n_pos = int(pos_mask.sum().item())
        n_neg = int(neg_mask.sum().item())

        # No positives (or no negatives) in this batch: fall back to BCE so the
        # gradient step is still useful and logits remain calibrated.
        if n_pos == 0 or n_neg == 0:
            return Fn.binary_cross_entropy_with_logits(logits, targets)

        s_pos = scores[pos_mask]           # (P,)
        s_neg = scores[neg_mask]           # (N,)
        # all P*N pairwise differences (s_pos - s_neg)
        diff = s_pos.unsqueeze(1) - s_neg.unsqueeze(0)   # (P, N)
        hinge = torch.clamp(self.margin - diff, min=0.0)
        loss = (hinge * hinge).mean()

        if self.bce_anchor > 0.0:
            loss = loss + self.bce_anchor * Fn.binary_cross_entropy_with_logits(
                logits, targets
            )
        return loss


def make_loss(cfg, *, n_pos: Optional[int] = None, n_neg: Optional[int] = None) -> nn.Module:
    """Build the training loss from a config block.

    ``cfg`` is the ``vision.loss`` sub-config (a dict / Config), e.g.::

        loss:
          name: bce              # bce | focal | weighted_bce | pauc_surrogate
          label_smoothing: 0.05  # bce / weighted_bce
          pos_weight: null       # bce: null -> 1.0 ; weighted_bce: null -> auto n_neg/n_pos
          alpha: 0.9             # focal only
          gamma: 2.0             # focal only
          margin: 1.0            # pauc_surrogate only
          on_logits: false       # pauc_surrogate only
          bce_anchor: 0.05       # pauc_surrogate only

    ``n_pos``/``n_neg`` are the per-fold train-side class counts, used only to
    derive the auto ``pos_weight`` for the weighted_bce alias when unspecified.

    Default is "bce" (plain BCE + label smoothing 0.05), the proven recipe once
    the sampler balances the batch. Returns an ``nn.Module`` (move it to the
    device with ``.to(device)``).
    """
    # accept dict / Config / None; fall back to the proven BCE+LS default
    get = (cfg.get if hasattr(cfg, "get") else (lambda k, d=None: d))
    name = str(get("name", "bce")).lower()

    if name == "focal":
        return FocalLoss(alpha=float(get("alpha", 0.9)),
                         gamma=float(get("gamma", 2.0)))

    if name in ("bce", "weighted_bce", "wbce"):
        ls = float(get("label_smoothing", 0.05))
        pw = get("pos_weight", None)
        if pw is None:
            # plain "bce": pos_weight 1.0 (sampler handles balance). The
            # "weighted_bce" alias keeps the old auto n_neg/n_pos up-weighting.
            if name in ("weighted_bce", "wbce") and n_pos and n_neg:
                pw = float(n_neg) / float(max(n_pos, 1))
            else:
                pw = 1.0
        return BCELoss(pos_weight=float(pw), label_smoothing=ls)

    if name in ("pauc_surrogate", "pauc", "auc_margin", "surrogate"):
        return PAUCSurrogateLoss(
            margin=float(get("margin", 1.0)),
            on_logits=bool(get("on_logits", False)),
            bce_anchor=float(get("bce_anchor", 0.05)),
        )

    raise ValueError(
        f"unknown loss name {name!r}; choose bce | focal | weighted_bce | pauc_surrogate"
    )
