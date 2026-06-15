"""Efficiency measurement — the second axis of the whole project.

Every model that reports a pAUC number must also report (params, FLOPs, CPU
latency) so each appears as one point on the quality-vs-cost frontier. These
three are measured consistently here so frontier points are comparable.

CPU-LATENCY HONESTY (why this file was rewritten)
-------------------------------------------------
The original ``cpu_latency_ms`` ran inline at the end of every training fold,
on a box that was simultaneously running a GPU training sweep. With the default
20-thread CPU pool, a single forward contends with the training process for
cores, and the timing is dominated by scheduler jitter rather than the model.
That is how ``reports/frontier.csv`` ended up physically impossible: 128px
backbones logged ~800-1200 ms while 224/256px ones logged 13-48 ms (a bigger
input at higher resolution cannot be 50x cheaper). Confirmed empirically: a
clean single-thread re-measure of ``mnv4_small`` @128 is ~2.8 ms, not 959 ms.

The fix here:
  * pin the thread count (default ``threads=1``) so the number is deterministic
    and reproducible, and is the honest "one core, one image" telederm story;
    a thread sweep is available via ``cpu_latency_ms(..., threads=None)`` (uses
    whatever the process default is) for the multi-core curiosity, but the
    frontier table is built at a fixed, documented thread count.
  * ``model.eval()`` + ``torch.inference_mode()`` (no autograd graph).
  * warmup forwards before timing; median (not mean) of N single-image forwards
    at the model's ACTUAL ``img_size``.
  * for very large models ``measure`` auto-shrinks N so the table still finishes
    on CPU.
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import numpy as np

# Fixed thread count for the AUTHORITATIVE cost table. 1 == "one core, one
# image", the honest resource-constrained / telederm deployment story, and the
# most reproducible single number across machines. Documented in the CSV-builder
# and in the frontier figure caption.
DEFAULT_THREADS = 1


@dataclass
class Cost:
    params_m: float          # millions of parameters
    gflops: float            # GFLOPs for a single 1x3xHxW forward
    cpu_ms: float            # median CPU latency per image, milliseconds
    img_size: int

    def as_row(self) -> dict:
        return asdict(self)


def count_params(model) -> float:
    return sum(p.numel() for p in model.parameters()) / 1e6


def count_flops(model, img_size: int = 128) -> float:
    """GFLOPs via fvcore if available, else thop, else NaN (logged, not fatal)."""
    import torch

    x = torch.randn(1, 3, img_size, img_size)
    model = model.eval()
    with torch.inference_mode():
        try:
            from fvcore.nn import FlopCountAnalysis

            fca = FlopCountAnalysis(model, x)
            fca.unsupported_ops_warnings(False)
            fca.uncalled_modules_warnings(False)
            return float(fca.total()) / 1e9
        except Exception:
            try:
                from thop import profile

                macs, _ = profile(model, inputs=(x,), verbose=False)
                return float(macs) * 2 / 1e9  # MACs -> FLOPs
            except Exception:
                return float("nan")


def cpu_latency_ms(model, img_size: int = 128, n: int = 50, warmup: int = 8,
                   threads: int | None = DEFAULT_THREADS) -> float:
    """Median single-image CPU forward latency, in ms.

    CPU because the deployment story here is resource-constrained / telederm
    (mirrors the competition CPU prize). Deterministic and honest:

      * ``model.eval()`` + ``.cpu()`` + ``torch.inference_mode()``.
      * ``threads`` pins ``torch.set_num_threads`` for the duration (restored on
        exit). ``threads=1`` is the authoritative single-core number; pass
        ``threads=None`` to time at the process default. Pinning is what makes a
        128px-vs-256px comparison monotonic instead of jitter-dominated.
      * ``warmup`` untimed forwards, then median of ``n`` timed single-image
        forwards at the model's actual ``img_size``.

    Never raises: any failure (e.g. a backbone that won't run on CPU) -> NaN.
    """
    import torch

    prev_threads = torch.get_num_threads()
    try:
        if threads is not None:
            torch.set_num_threads(int(threads))
        model = model.eval().cpu()
        x = torch.randn(1, 3, img_size, img_size)
        with torch.inference_mode():
            for _ in range(warmup):
                model(x)
            ts = []
            for _ in range(n):
                t0 = time.perf_counter()
                model(x)
                ts.append((time.perf_counter() - t0) * 1e3)
        return float(np.median(ts))
    except Exception:
        return float("nan")
    finally:
        torch.set_num_threads(prev_threads)


def _latency_n(params_m: float) -> int:
    """Auto-shrink the number of timed forwards for heavy models so the cost
    table still finishes on CPU. mnv5_300m (~300M) / eva02@336 are slow per
    forward; 10 medianed samples are plenty and stable."""
    if not np.isfinite(params_m):
        return 20
    if params_m >= 100:      # mnv5_300m and friends
        return 10
    if params_m >= 25:       # convnextv2_tiny / swinv2_tiny / vit_small
        return 20
    return 50                # small/cheap backbones


def measure(model, img_size: int = 128, threads: int | None = DEFAULT_THREADS) -> Cost:
    """One frontier point: (params, FLOPs, CPU latency) at ``img_size``.

    N for the latency median auto-scales down with model size so heavy anchors
    (mnv5_300m, eva02@336) don't make the table take forever on CPU.
    """
    p = count_params(model)
    return Cost(
        params_m=round(p, 3),
        gflops=round(count_flops(model, img_size), 4),
        cpu_ms=round(cpu_latency_ms(model, img_size,
                                    n=_latency_n(p), threads=threads), 3),
        img_size=img_size,
    )


# --- tabular (LightGBM) cost -------------------------------------------------

def _to_lgb_booster(model):
    """Reach the raw LightGBM Booster through the common wrapper layers.

    The persisted GBDT ensemble holds ``gbdt._RankModel`` adapters; the LightGBM
    ones wrap an sklearn ``LGBMClassifier`` whose ``.booster_`` is the Booster.
    Returns the Booster, or None if this object is not a LightGBM model
    (e.g. a CatBoost ``_RankModel``)."""
    m = model
    m = getattr(m, "model", m)          # unwrap _RankModel
    if hasattr(m, "booster_"):           # sklearn LGBMClassifier / LGBMRegressor
        return m.booster_
    if hasattr(m, "trees_to_dataframe") or hasattr(m, "dump_model"):
        return m                         # already a raw Booster
    return None


def _booster_leaf_count(model) -> int:
    """Total number of leaves in one GBDT model (LightGBM or CatBoost).

    Leaves are the right "parameter" proxy for a GBDT: each leaf stores one
    learned output value, so the leaf count is the model's free-parameter
    budget the same way `numel()` is for a neural net. We sum across every
    tree (and across folds/seeds in the ensemble) to mirror how the NN side
    sums params across constituent models for an ensemble frontier point.

    Handles three shapes: a ``_RankModel`` adapter, an sklearn LightGBM
    classifier, or a raw LightGBM Booster -- and CatBoost (symmetric trees,
    so leaves = 2**depth per tree).
    """
    booster = _to_lgb_booster(model)
    if booster is not None:
        # LightGBM. Fast path: trees_to_dataframe() has one row per node; leaves
        # have no split feature. Falls back to the JSON dump if unavailable.
        try:
            tdf = booster.trees_to_dataframe()
            if "split_feature" in tdf.columns:
                return int(tdf["split_feature"].isna().sum())
            if "leaf_value" in tdf.columns:
                return int(tdf["leaf_value"].notna().sum())
        except Exception:
            pass

        def _count(node) -> int:
            if "leaf_value" in node or "leaf_index" in node:
                return 1
            total = 0
            for key in ("left_child", "right_child"):
                child = node.get(key)
                if isinstance(child, dict):
                    total += _count(child)
            return total

        dump = booster.dump_model()
        return sum(_count(t["tree_structure"]) for t in dump.get("tree_info", []))

    # CatBoost: oblivious/symmetric trees, every tree has exactly 2**depth leaves.
    cat = getattr(model, "model", model)
    try:
        n_trees = int(cat.tree_count_)
        depth = int(cat.get_all_params().get("depth", 6))
        return n_trees * (2 ** depth)
    except Exception:
        return 0


def measure_gbdt(boosters, X_sample) -> dict:
    """Cost of a LightGBM (per-fold) ensemble, in the same schema as
    ``Cost.as_row()`` so a tabular point lines up column-for-column with the
    neural-net frontier points.

    Returns ``{params_m, gflops, cpu_ms, img_size}`` where:
      * params_m -- total leaves across all per-fold boosters / 1e6 (leaves are
        the GBDT's learned parameters; see ``_booster_leaf_count``).
      * gflops   -- 0.0 (a tree ensemble does no dense floating-point matmul;
        cost is comparisons/lookups, not FLOPs). Plotted as a tabular
        reference rather than on the log-FLOP curve.
      * cpu_ms   -- median per-ROW predict latency in ms, timed on ``X_sample``.
      * img_size -- 0 (tabular model takes no image).

    ``boosters`` may be a list of raw boosters or of ``(booster, cols, state)``
    tuples (the shape produced by ``gbdt.train_oof``). Never raises: timing
    failures yield ``cpu_ms = NaN``; leaf-count failures yield ``params_m =
    NaN`` so the point is logged but flagged rather than silently wrong.
    """
    blist = list(boosters) if boosters is not None else []

    def _unpack(b):
        """-> (booster, cols_or_None) for either tuple or bare-booster shapes."""
        if isinstance(b, (tuple, list)):
            booster = b[0]
            cols = b[1] if len(b) > 1 else None
            return booster, cols
        return b, None

    # --- params_m : total leaves / 1e6 ---
    try:
        total_leaves = 0
        for b in blist:
            booster, _ = _unpack(b)
            total_leaves += _booster_leaf_count(booster)
        params_m = round(total_leaves / 1e6, 6)
    except Exception:
        params_m = float("nan")

    # --- cpu_ms : median per-row predict latency, timed on the first booster ---
    cpu_ms = float("nan")
    try:
        if blist:
            booster, cols = _unpack(blist[0])
            X = X_sample
            if cols is not None:
                try:
                    X = X_sample[list(cols)]
                except Exception:
                    X = X_sample  # fall back to the full frame on column mismatch
            n_rows = max(int(getattr(X, "shape", [1])[0]), 1)

            # warmup (also primes any lazy LightGBM state)
            for _ in range(2):
                booster.predict(X)

            n_reps = 7
            per_row = []
            for _ in range(n_reps):
                t0 = time.perf_counter()
                booster.predict(X)
                per_row.append((time.perf_counter() - t0) * 1e3 / n_rows)
            cpu_ms = round(float(np.median(per_row)), 5)
    except Exception:
        cpu_ms = float("nan")

    return {"params_m": params_m, "gflops": 0.0, "cpu_ms": cpu_ms, "img_size": 0}
