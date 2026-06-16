"""Quality-vs-cost Pareto frontier for ISIC-2024 SLICE-3D (efficiency axis).

Two jobs, one file:

  1. ``--build-cost``  Re-measure the AUTHORITATIVE cost of every backbone in
     ``src/vision/backbones.py:FRONTIER`` at its INTENDED img_size (read from
     ``configs/vision/<name>.yaml`` when present, else 128px), plus the tabular
     GBDT ensemble, and write ``reports/frontier_cost.csv``.

     Why this exists: the in-run ``efficiency.measure`` ran inline at the end of
     every training fold, on a box simultaneously running a GPU sweep, with the
     default 20-thread CPU pool. The CPU latency that produced was scheduler
     jitter, not the model -- 128px backbones logged ~800-1200 ms while 224/256px
     ones logged 13-48 ms (a bigger input cannot be 50x cheaper). The authoritative
     table re-measures cleanly, single-threaded, in isolation. Costs are the honest
     "one core, one image" telederm story.

  2. (default)  Plot the frontier. The AUTHORITATIVE cost (frontier_cost.csv) is
     JOINED with the best (max) OOF pAUC per model from ``reports/frontier.csv``
     (which has duplicate rows -- several 'gbdt', etc). Three panels are produced:
     pAUC vs params_m, vs GFLOPs, vs CPU latency (ms), each on a log-x axis, plus a
     combined 1x3 figure. The Pareto-optimal set (max pAUC at <= cost) is ringed and
     connected; the tabular GBDT pAUC is drawn as a dashed reference line; stacked
     ensemble points are marked distinctly.

CPU LATENCY is measured single-threaded (``efficiency.DEFAULT_THREADS``); see the
figure caption / this docstring. Everything is robust to missing rows: a backbone
that fails to build logs NaN cost and is skipped on the plot, never crashing it.
"""
from __future__ import annotations

import argparse
import glob
import os

import matplotlib
import numpy as np
import pandas as pd
import yaml

matplotlib.use("Agg")  # headless: run in CI / on a box with no display
import matplotlib.pyplot as plt  # noqa: E402

from reports import _style  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST_CSV = os.path.join(REPO, "reports", "frontier_cost.csv")
PAUC_CSV = os.path.join(REPO, "reports", "frontier.csv")
CFG_DIR = os.path.join(REPO, "configs", "vision")
GBDT_BOOSTERS = os.path.join(REPO, "experiments", "gbdt_boosters.joblib")

# Frontier rows that are STACKED ensembles (image expert(s) + tabular), not a
# single backbone. Their cost is the sum across constituents; we don't re-measure
# them here (they have no single img_size), so on the plot they carry their
# in-CSV cost but are marked distinctly. Detected by name substring.
STACK_TOKENS = ("stack", "+")


# --------------------------------------------------------------------------- #
# 1) build the authoritative cost table
# --------------------------------------------------------------------------- #
def _config_img_size() -> dict:
    """Map intended frontier-row NAME -> img_size, read from configs/vision/*.yaml.

    The row name is the OOF tag (top-level ``name`` in the config, e.g.
    ``convnextv2_nano_r224``) so a re-resolution variant of the same backbone is
    a DISTINCT cost point. Falls back to the ``backbone`` field for configs that
    don't override ``name``.
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(CFG_DIR, "*.yaml"))):
        try:
            d = yaml.safe_load(open(f)) or {}
        except Exception:
            continue
        # configs are wrapped under a top-level ``vision:`` key
        v = d.get("vision", d)
        backbone = v.get("backbone")
        if backbone is None:
            continue
        img = int(v.get("img_size", 128))
        name = v.get("name") or backbone
        out[name] = {"backbone": backbone, "img_size": img}
    return out


def build_cost_table() -> pd.DataFrame:
    """Re-measure every FRONTIER backbone at its intended img_size + the GBDT.

    Each backbone is built ``pretrained=False`` (fast, no download). A build or
    measure failure logs NaN cost and continues (never crashes the table).
    """
    from src import efficiency as eff
    from src.vision import backbones

    cfg = _config_img_size()
    # Build a {name -> (backbone, img_size)} plan: every config-named row, PLUS
    # any backbone in FRONTIER that has no config (default 128px).
    plan: dict[str, dict] = dict(cfg)
    configured_backbones = {v["backbone"] for v in cfg.values()}
    for bname in backbones.FRONTIER:
        if bname not in plan and bname not in configured_backbones:
            plan[bname] = {"backbone": bname, "img_size": 128}

    print(f"[build-cost] threads={eff.DEFAULT_THREADS}  "
          f"{len(plan)} backbone points + gbdt")

    rows = []
    for name in sorted(plan):
        backbone = plan[name]["backbone"]
        img = plan[name]["img_size"]
        try:
            model = backbones.build(backbone, pretrained=False, img_size=img)
            cost = eff.measure(model, img)
            r = cost.as_row()
            r["model"] = name
            rows.append(r)
            print(f"  {name:24s} img={img:<4d} "
                  f"params={r['params_m']:.3f}M  gflops={r['gflops']:.4f}  "
                  f"cpu_ms={r['cpu_ms']:.3f}")
            del model
        except Exception as e:  # noqa: BLE001 - log NaN cost and keep going
            rows.append({"model": name, "params_m": float("nan"),
                         "gflops": float("nan"), "cpu_ms": float("nan"),
                         "img_size": img})
            print(f"  {name:24s} img={img:<4d} BUILD FAILED ({type(e).__name__}: "
                  f"{str(e)[:80]}) -> NaN cost")

    # --- tabular GBDT ---
    try:
        import joblib
        if os.path.exists(GBDT_BOOSTERS):
            boosters = joblib.load(GBDT_BOOSTERS)
            cols = None
            b0 = boosters[0]
            if isinstance(b0, (tuple, list)) and len(b0) > 1:
                cols = list(b0[1])
            n_feat = len(cols) if cols else 64
            rng = np.random.default_rng(42)
            X = pd.DataFrame(rng.standard_normal((256, n_feat)).astype("float32"),
                             columns=cols if cols else None)
            g = eff.measure_gbdt(boosters, X)
            g["model"] = "gbdt"
            rows.append(g)
            print(f"  {'gbdt':24s}        "
                  f"params={g['params_m']}M  gflops={g['gflops']}  "
                  f"cpu_ms={g['cpu_ms']}")
        else:
            rows.append({"model": "gbdt", "params_m": 0.0, "gflops": 0.0,
                         "cpu_ms": 0.001, "img_size": 0})
            print("  gbdt: boosters not found -> placeholder cost")
    except Exception as e:  # noqa: BLE001
        rows.append({"model": "gbdt", "params_m": float("nan"), "gflops": 0.0,
                     "cpu_ms": float("nan"), "img_size": 0})
        print(f"  gbdt: measure failed ({type(e).__name__}: {str(e)[:80]}) -> NaN")

    df = pd.DataFrame(rows)[["model", "params_m", "gflops", "cpu_ms", "img_size"]]
    df.to_csv(COST_CSV, index=False)
    print(f"[build-cost] wrote {COST_CSV} ({len(df)} rows)")
    return df


# --------------------------------------------------------------------------- #
# 2) pareto helpers
# --------------------------------------------------------------------------- #
def pareto_front(df: pd.DataFrame, cost: str, quality: str = "pauc") -> pd.DataFrame:
    """Lower-cost, higher-quality Pareto front. Expects finite, plottable cost."""
    d = df.sort_values(cost)
    keep, best = [], -np.inf
    for _, r in d.iterrows():
        if r[quality] > best:
            keep.append(r)
            best = r[quality]
    return pd.DataFrame(keep)


def is_pareto(df: pd.DataFrame, cost: str, quality: str = "pauc") -> pd.Series:
    """Boolean mask: True iff the row is Pareto-optimal (nothing both <= cost
    and >= quality dominates it)."""
    mask = pd.Series(False, index=df.index)
    if len(df) == 0:
        return mask
    c = df[cost].to_numpy(dtype=float)
    q = df[quality].to_numpy(dtype=float)
    for i in range(len(df)):
        dominated = False
        for j in range(len(df)):
            if j == i:
                continue
            if c[j] <= c[i] and q[j] >= q[i] and (c[j] < c[i] or q[j] > q[i]):
                dominated = True
                break
        mask.iloc[i] = not dominated
    return mask


# --------------------------------------------------------------------------- #
# 3) join cost + pAUC
# --------------------------------------------------------------------------- #
def _is_stack(name: str) -> bool:
    s = str(name).lower()
    return any(tok in s for tok in STACK_TOKENS)


# The "3img" stack convention in experiments/run_stack.py = these three 128px
# image experts (+ the tabular GBDT). We recompute a stack point's cost as the
# SUM of its constituents' AUTHORITATIVE costs (the project rule: ensemble cost
# = sum over constituents), instead of trusting the polluted summed cost that
# run_stack.py read out of the buggy frontier.csv.
_THREE_IMG = ["convnextv2_nano", "vit_tiny", "mnv4_small"]


def _stack_constituents(name: str) -> list[str]:
    s = str(name).lower()
    members = []
    if "gbdt" in s or "lgbm" in s or "stack" in s:
        members.append("gbdt")
    if "3img" in s:
        members += _THREE_IMG
    else:  # single-image stacks name the backbone explicitly
        for bb in ("convnextv2_nano_r224", "convnextv2_nano", "convnextv2_tiny",
                   "vit_tiny", "vit_small", "mnv4_small", "swinv2_tiny",
                   "effnetv2_b0", "eva02_small", "mnv5_300m"):
            if bb in s and bb not in members:
                members.append(bb)
    return members


def _recompute_stack_costs(joined: pd.DataFrame, cost: pd.DataFrame) -> None:
    """In place: replace each stack row's cost with the sum of its constituents'
    authoritative costs. The udk/pca/gate add-ons are tabular-only post-features
    on top of the same backbones, so they add no backbone compute. Falls back to
    the existing (in-CSV) cost if a constituent cost is missing."""
    clut = cost.set_index("model")
    for i in joined.index:
        name = joined.at[i, "model"]
        if not _is_stack(name):
            continue
        if name in clut.index:  # authoritative cost already in frontier_cost.csv -> trust it
            continue
        members = _stack_constituents(name)
        present = [m for m in members if m in clut.index]
        if not present:
            continue  # keep in-CSV fallback
        for c in ("params_m", "gflops", "cpu_ms"):
            vals = pd.to_numeric(clut.loc[present, c], errors="coerce")
            if vals.notna().any():
                joined.at[i, c] = float(vals.sum(skipna=True))


def load_joined() -> tuple[pd.DataFrame, float | None]:
    """Best (max) pAUC per model from frontier.csv, joined to authoritative cost.

    Returns ``(joined_df, gbdt_pauc)``. ``joined_df`` has one row per model with
    columns ``model, pauc, params_m, gflops, cpu_ms, img_size, is_stack``. The
    tabular GBDT pAUC is returned separately as the reference line and excluded
    from the scatter (no FLOPs / no image -> categorically different cost).
    """
    # --- best pAUC per model (dedupe the messy frontier.csv) ---
    try:
        pa = pd.read_csv(PAUC_CSV)
        pa = pa.loc[:, ~pa.columns.duplicated()]
    except (FileNotFoundError, pd.errors.EmptyDataError):
        pa = pd.DataFrame(columns=["model", "pauc"])
    pa["pauc"] = pd.to_numeric(pa.get("pauc"), errors="coerce")
    pa["model"] = pa.get("model", pd.Series(dtype=str)).astype(str)
    best_pauc = (pa.dropna(subset=["pauc"]).groupby("model", as_index=False)["pauc"]
                 .max())

    # --- authoritative cost ---
    try:
        cost = pd.read_csv(COST_CSV)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        cost = pd.DataFrame(columns=["model", "params_m", "gflops", "cpu_ms",
                                     "img_size"])
    cost["model"] = cost.get("model", pd.Series(dtype=str)).astype(str)
    for c in ("params_m", "gflops", "cpu_ms", "img_size"):
        cost[c] = pd.to_numeric(cost.get(c), errors="coerce")

    # Join on model name. Inner-ish: keep every model that has a pAUC; bring in
    # authoritative cost where we have it. Stacked points have NO authoritative
    # cost row -> fall back to their in-CSV cost so they still plot.
    joined = best_pauc.merge(cost, on="model", how="left")

    # backfill stack costs from the raw frontier.csv (summed-constituent cost)
    raw_cost_cols = {}
    for c in ("params_m", "gflops", "cpu_ms", "img_size"):
        if c in pa.columns:
            raw_cost_cols[c] = pd.to_numeric(pa[c], errors="coerce")
    if raw_cost_cols:
        raw = pd.DataFrame({"model": pa["model"], **raw_cost_cols})
        # for each model, take the max-pauc row's cost as the in-csv fallback
        idx = pa.dropna(subset=["pauc"]).groupby("model")["pauc"].idxmax()
        fallback = raw.loc[idx].set_index("model")
        for m in joined["model"]:
            row = joined["model"] == m
            for c in ("params_m", "gflops", "cpu_ms", "img_size"):
                if c in fallback.columns and joined.loc[row, c].isna().all() \
                        and m in fallback.index:
                    joined.loc[row, c] = fallback.loc[m, c]

    joined["is_stack"] = joined["model"].map(_is_stack)
    # ensemble cost = sum of constituents' AUTHORITATIVE costs (overrides the
    # polluted summed cost run_stack.py copied out of the buggy frontier.csv)
    _recompute_stack_costs(joined, cost)

    gbdt_pauc = None
    gmask = joined["model"].str.lower().isin(["gbdt", "lightgbm", "lgbm",
                                              "tabular"])
    if gmask.any():
        gbdt_pauc = float(joined.loc[gmask, "pauc"].max())
    scatter = joined[~gmask].copy()
    return scatter, gbdt_pauc


# --------------------------------------------------------------------------- #
# 4) plotting
# --------------------------------------------------------------------------- #
DPI = 300
_style.apply(plt, dpi=DPI)
_AXLABEL = {"params_m": "Parameters (M)", "gflops": "GFLOPs",
            "cpu_ms": "CPU latency (ms, single-thread)"}
C_IMG = _style.CB["blue"]
C_STK = _style.CB["orange"]
C_PAR = _style.CB["red"]
C_GBD = _style.CB["green"]


# Per-model leader-line label placement (dx, dy in points; ha, va), tuned so no
# label overlaps a marker, the GBDT line, the legend, or another label.
_LABEL_POS = {
    "mnv4_small": (0, -20, "center", "top"),
    "effvit_b0": (0, 18, "center", "bottom"),
    "vit_tiny": (0, 20, "center", "bottom"),
    "convnextv2_nano": (-46, 18, "right", "bottom"),
    "convnextv2_nano_r224": (0, -20, "center", "top"),
    "convnextv2_tiny": (46, 6, "left", "center"),
    "swinv2_tiny": (0, 18, "center", "bottom"),
    "eva02_small": (0, 18, "center", "bottom"),
    "stack_best": (-10, 22, "right", "bottom"),
    "stack_gbdt+3img+udk": (40, -16, "left", "top"),
}
_DEFAULT_POS = (0, 18, "center", "bottom")


def _plot_axis(ax, df: pd.DataFrame, cost: str, gbdt_pauc: float | None,
               annotate: bool = True):
    plottable = df[(df[cost] > 0) & np.isfinite(df[cost]) & df["pauc"].notna()].copy()
    dropped = df[~df.index.isin(plottable.index)]

    if gbdt_pauc is not None:
        ax.axhline(gbdt_pauc, ls="--", lw=1.6, color=C_GBD, alpha=0.95,
                   label=f"LightGBM tabular: {gbdt_pauc:.4f}")

    if len(plottable):
        ax.set_xscale("log")
        single = plottable[~plottable["is_stack"]]
        stacks = plottable[plottable["is_stack"]]
        pareto_mask = is_pareto(plottable, cost)
        front = pareto_front(plottable, cost)

        if len(single):
            ax.scatter(single[cost], single["pauc"], s=70, alpha=0.85,
                       color=C_IMG, label="image expert", zorder=2)
        if len(stacks):
            ax.scatter(stacks[cost], stacks["pauc"], s=120, alpha=0.9,
                       marker="D", color=C_STK, label="stacked ensemble", zorder=3)
        ax.plot(front[cost], front["pauc"], "-", color=C_PAR, lw=2.0,
                zorder=4, label="Pareto front")
        po = plottable[pareto_mask]
        ax.scatter(po[cost], po["pauc"], s=200, facecolors="none",
                   edgecolors=C_PAR, linewidths=2.0, zorder=5)

        if annotate:
            for _, r in plottable.iterrows():
                dx, dy, ha, va = _LABEL_POS.get(str(r["model"]), _DEFAULT_POS)
                ax.annotate(
                    str(r["model"]), (r[cost], r["pauc"]),
                    fontsize=11, zorder=6,
                    xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va,
                    arrowprops=dict(arrowstyle="-", color="#888", lw=0.7,
                                    shrinkA=0, shrinkB=3))
    else:
        ax.text(0.5, 0.5, "no costed runs for this axis",
                ha="center", va="center", transform=ax.transAxes,
                fontsize=13, color="gray")

    ax.set_xlabel(_AXLABEL.get(cost, cost) + " (log scale)")
    ax.set_ylabel("OOF pAUC @ 80% TPR")
    ax.margins(y=0.22)
    ax.grid(True, which="both", ls=":", alpha=0.35)
    ax.legend(loc="upper left", framealpha=0.92)
    return plottable, dropped


def make_figures():
    df, gbdt_pauc = load_joined()
    axes_files = {
        "params_m": os.path.join(REPO, "reports", "frontier_params"),
        "gflops": os.path.join(REPO, "reports", "frontier_gflops"),
        "cpu_ms": os.path.join(REPO, "reports", "frontier_cpu"),
    }

    for cost, base in axes_files.items():
        fig, ax = plt.subplots(figsize=(9.0, 6.6), constrained_layout=True)
        _plot_axis(ax, df, cost, gbdt_pauc)
        ax.set_title(f"SLICE-3D: pAUC vs {_AXLABEL.get(cost, cost)}\n"
                     "(single-dataset, no external data)")
        _style.save_vector(fig, base, dpi=DPI)
        plt.close(fig)
        print(f"wrote {base}.{{svg,png}}")

    combined = os.path.join(REPO, "reports", "frontier")
    fig, axs = plt.subplots(1, 3, figsize=(21, 6.8), constrained_layout=True)
    for k, (ax, cost) in enumerate(zip(axs, ("params_m", "gflops", "cpu_ms"))):
        _plot_axis(ax, df, cost, gbdt_pauc, annotate=True)
        ax.set_title(_AXLABEL.get(cost, cost))
        _style.panel_label(ax, "abc"[k])
    fig.suptitle("ISIC-2024 SLICE-3D quality-vs-cost frontier "
                 "(authoritative single-thread cost; single-dataset, no external data)")
    _style.save_vector(fig, combined, dpi=DPI)
    plt.close(fig)
    print(f"wrote {combined}.{{svg,png}}")

    return df, gbdt_pauc


def _report(df: pd.DataFrame, gbdt_pauc: float | None):
    print("\n=== deduped pAUC-vs-cost table (best pAUC per model) ===")
    show = df.copy()
    cols = ["model", "pauc", "params_m", "gflops", "cpu_ms", "img_size",
            "is_stack"]
    show = show[[c for c in cols if c in show.columns]]
    show = show.sort_values("pauc", ascending=False)
    with pd.option_context("display.float_format", lambda v: f"{v:.4f}",
                           "display.width", 160, "display.max_columns", 20):
        print(show.to_string(index=False))
    if gbdt_pauc is not None:
        print(f"\ntabular GBDT reference pAUC = {gbdt_pauc:.4f}")

    for cost in ("params_m", "gflops", "cpu_ms"):
        plottable = df[(df[cost] > 0) & np.isfinite(df[cost])
                       & df["pauc"].notna()]
        if not len(plottable):
            print(f"\n[Pareto on {cost}] no plottable points")
            continue
        po = plottable[is_pareto(plottable, cost)].sort_values(cost)
        names = ", ".join(f"{r['model']}({r[cost]:.3g}, pAUC {r['pauc']:.4f})"
                          for _, r in po.iterrows())
        print(f"\n[Pareto-optimal on {cost}] {names}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-cost", action="store_true",
                    help="re-measure all backbones + GBDT -> reports/frontier_cost.csv")
    ap.add_argument("--no-plot", action="store_true",
                    help="with --build-cost, skip the figures")
    a = ap.parse_args()

    if a.build_cost:
        build_cost_table()
        if a.no_plot:
            return

    df, gbdt_pauc = make_figures()
    _report(df, gbdt_pauc)


if __name__ == "__main__":
    main()
