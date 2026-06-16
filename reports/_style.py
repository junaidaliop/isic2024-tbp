"""Shared publication-grade matplotlib style for all ISIC-2024 report figures."""
from __future__ import annotations

import numpy as np
from matplotlib import font_manager

# Font: prefer IBM Plex Sans (deck), else Arial/Helvetica, else DejaVu Sans.
_AVAIL = {f.name for f in font_manager.fontManager.ttflist}
for _f in ("IBM Plex Sans", "Arial", "Helvetica", "DejaVu Sans"):
    if _f in _AVAIL:
        FONT = _f
        break
else:
    FONT = "DejaVu Sans"

# Okabe-Ito colorblind-safe palette.
CB = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "black": "#000000",
    "grey": "#999999",
}

RC = {
    "font.family": FONT,
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 16,
    "figure.titleweight": "bold",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
}


def apply(plt, dpi=300):
    rc = dict(RC)
    rc["figure.dpi"] = dpi
    rc["savefig.dpi"] = dpi
    plt.rcParams.update(rc)


def panel_label(ax, letter, x=-0.02, y=1.06):
    ax.text(x, y, f"({letter})", transform=ax.transAxes, fontsize=15,
            fontweight="bold", va="bottom", ha="right")


def save_vector(fig, base, dpi=300):
    fig.savefig(base + ".svg")
    fig.savefig(base + ".png", dpi=dpi)


def save_raster(fig, base, dpi=300):
    fig.savefig(base + ".png", dpi=dpi)


def is_clean_crop(arr, black_thresh=20, black_frac_max=0.03, min_std=6.0):
    """Reject crops with a black padding bar (>3% near-black pixels) or near-uniform content."""
    a = np.asarray(arr)
    if a.ndim != 3:
        return False
    mx = a.max(axis=2)
    if (mx < black_thresh).mean() > black_frac_max:
        return False
    if float(a.std()) < min_std:
        return False
    return True


def select_clean(loader, candidate_ids, n):
    out = []
    for iid in candidate_ids:
        try:
            arr = loader(iid)
        except Exception:
            continue
        if is_clean_crop(arr):
            out.append(iid)
        if len(out) == n:
            break
    return out
