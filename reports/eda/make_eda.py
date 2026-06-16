"""Publication-grade EDA for ISIC-2024 SLICE-3D.

Read-only on data/. Writes reports/eda/*.{svg,png} and _stats.json.
Run: PYTHONPATH=. python reports/eda/make_eda.py
"""
from __future__ import annotations

import json
import os
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from reports import _style

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = "/home/efreet/Ali/isic2024-tbp"
OUT = os.path.join(ROOT, "reports", "eda")
os.makedirs(OUT, exist_ok=True)

DPI = 300
CB = _style.CB
BENIGN = CB["sky"]
MALIG = CB["red"]
_style.apply(plt, dpi=DPI)


def save_vec(fig, name):
    base = os.path.join(OUT, name)
    _style.save_vector(fig, base, dpi=DPI)
    plt.close(fig)
    print("wrote", base + ".{svg,png}")
    return base + ".png"


def save_png(fig, name):
    base = os.path.join(OUT, name)
    _style.save_raster(fig, base, dpi=DPI)
    plt.close(fig)
    print("wrote", base + ".png")
    return base + ".png"


print("loading data ...")
df = pd.read_csv(os.path.join(ROOT, "data", "train-metadata.csv"), low_memory=False)
folds = pd.read_parquet(os.path.join(ROOT, "data", "folds.parquet"))
ben = df[df.target == 0]
mal = df[df.target == 1]
N = len(df)
NPOS = int(df.target.sum())
PREV = 100 * df.target.mean()
NPAT = df.patient_id.nunique()

paths = {}

# --- FIG 1: class imbalance + per-fold positives ---
fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.4), constrained_layout=True)
counts = [N - NPOS, NPOS]
bars = ax[0].bar(["Benign", "Malignant"], counts, color=[BENIGN, MALIG], width=0.6)
ax[0].set_yscale("log")
ax[0].set_ylabel("Number of lesions (log scale)")
ax[0].set_title("Class imbalance")
for b, c in zip(bars, counts):
    ax[0].text(b.get_x() + b.get_width() / 2, c * 1.3, f"{c:,}",
               ha="center", va="bottom", fontweight="bold", fontsize=13)
ax[0].set_ylim(1, N * 12)
ax[0].text(0.97, 0.95,
           f"Prevalence = {PREV:.3f}%\n(1 malignant per {int(round(N / NPOS)):,})",
           transform=ax[0].transAxes, ha="right", va="top", fontsize=12,
           bbox=dict(boxstyle="round,pad=0.45", fc=CB["yellow"], ec="black", alpha=0.9))
_style.panel_label(ax[0], "a")

fg = folds.groupby("fold")["target"].agg(["count", "sum"])
fg["rate"] = 100 * fg["sum"] / fg["count"]
fb = ax[1].bar(fg.index.astype(str), fg["sum"], color=CB["blue"], width=0.6)
ax[1].set_xlabel("Cross-validation fold")
ax[1].set_ylabel("Malignant lesions in fold")
ax[1].set_title("Malignant count per fold")
for b, s, r in zip(fb, fg["sum"], fg["rate"]):
    ax[1].text(b.get_x() + b.get_width() / 2, s + 0.8, f"{int(s)}\n({r:.3f}%)",
               ha="center", va="bottom", fontsize=11)
ax[1].set_ylim(0, fg["sum"].max() * 1.28)
_style.panel_label(ax[1], "b")
paths["fig1_class_imbalance.png"] = save_vec(fig, "fig1_class_imbalance")

# --- FIG 2: patient structure ---
pp = df.groupby("patient_id").agg(n=("isic_id", "count"), mal=("target", "sum"))
n_mal_pat = int((pp.mal >= 1).sum())
mal_pat = pp[pp.mal >= 1]
nomal_pat = pp[pp.mal == 0]

fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.4), constrained_layout=True)
bins = np.logspace(0, np.log10(pp.n.max()), 40)
ax[0].hist(pp.n, bins=bins, color=CB["blue"], edgecolor="white", linewidth=0.4)
ax[0].set_xscale("log")
ax[0].set_xlabel("Lesions per patient (log scale)")
ax[0].set_ylabel("Number of patients")
ax[0].set_title("Lesions per patient")
ax[0].axvline(pp.n.median(), color=MALIG, ls="--", lw=2)
txt = (f"patients = {NPAT:,}\nmedian = {int(pp.n.median())} lesions\n"
       f"mean = {pp.n.mean():.0f} lesions\nmax = {int(pp.n.max()):,} lesions")
ax[0].text(0.97, 0.95, txt, transform=ax[0].transAxes, va="top", ha="right",
           fontsize=12,
           bbox=dict(boxstyle="round,pad=0.45", fc="white", ec=CB["grey"]))
_style.panel_label(ax[0], "a")

ax[1].bar(["No malignant\nlesion", "$\\geq$1 malignant\nlesion"],
          [len(nomal_pat), len(mal_pat)], color=[BENIGN, MALIG], width=0.55)
for i, v in enumerate([len(nomal_pat), len(mal_pat)]):
    ax[1].text(i, v + 8, f"{v}", ha="center", fontweight="bold", fontsize=13)
ax[1].set_ylabel("Number of patients")
ax[1].set_title(f"{n_mal_pat} of {NPAT} patients carry a malignant lesion")
ax[1].set_ylim(0, max(len(nomal_pat), len(mal_pat)) * 1.22)
ax[1].text(0.5, 0.70,
           f"these {n_mal_pat} patients hold\n{int(mal_pat.n.sum()):,} lesions "
           f"({100 * mal_pat.n.sum() / N:.0f}% of all crops)",
           transform=ax[1].transAxes, ha="center", fontsize=12,
           bbox=dict(boxstyle="round,pad=0.45", fc=CB["yellow"], ec="black", alpha=0.85))
_style.panel_label(ax[1], "b")
paths["fig2_patient_structure.png"] = save_vec(fig, "fig2_patient_structure")

# --- FIG 3: demographics ---
fig, ax = plt.subplots(1, 3, figsize=(17.5, 5.6), constrained_layout=True)
abins = np.arange(0, 91, 5)
ax[0].hist(ben.age_approx.dropna(), bins=abins, density=True, color=BENIGN,
           alpha=0.6, label="Benign", edgecolor="white", linewidth=0.3)
ax[0].hist(mal.age_approx.dropna(), bins=abins, density=True, histtype="step",
           color=MALIG, lw=2.5, label="Malignant")
ax[0].set_xlabel("Approximate age (years)")
ax[0].set_ylabel("Density")
ax[0].set_title("Age distribution by class")
ax[0].legend(loc="upper left")
_style.panel_label(ax[0], "a")

sx = df.groupby("sex")["target"].agg(["count", "sum", "mean"]).reindex(["female", "male"])
x = np.arange(2)
ax[1].bar(x, sx["count"], color=CB["grey"], width=0.55)
ax[1].set_xticks(x)
ax[1].set_xticklabels(["Female", "Male"])
ax[1].set_ylabel("Number of lesions")
ax[1].set_title("Lesion count by sex")
for i, (c, m) in enumerate(zip(sx["count"], sx["mean"])):
    ax[1].text(i, c + 4000, f"{int(c):,}\nmalig. {100 * m:.3f}%",
               ha="center", va="bottom", fontsize=11)
ax[1].set_ylim(0, sx["count"].max() * 1.26)
_style.panel_label(ax[1], "b")

st = (df.groupby("anatom_site_general")["target"].agg(["count", "mean"])
      .sort_values("mean"))
ax[2].barh(range(len(st)), 100 * st["mean"], color=CB["orange"], height=0.62)
ax[2].set_yticks(range(len(st)))
ax[2].set_yticklabels(list(st.index))
ax[2].set_xlabel("Malignant rate (%)")
ax[2].set_title("Malignant rate by anatomical site")
for i, (r, c) in enumerate(zip(st["mean"], st["count"])):
    ax[2].text(100 * r + 0.012, i, f"{100 * r:.3f}%  (n={int(c):,})",
               va="center", fontsize=11)
ax[2].set_xlim(0, 100 * st["mean"].max() * 1.55)
_style.panel_label(ax[2], "c")
paths["fig3_demographics.png"] = save_vec(fig, "fig3_demographics")

# --- FIG 4: lesion size ---
fig, ax = plt.subplots(1, 2, figsize=(13.5, 5.6), constrained_layout=True)
for k, (axi, col, label) in enumerate([
    (ax[0], "clin_size_long_diam_mm", "Clinical longest diameter (mm)"),
    (ax[1], "tbp_lv_areaMM2", "TBP lesion area (mm$^2$)"),
]):
    b = ben[col].dropna()
    m = mal[col].dropna()
    lo = max(min(b.min(), m.min()), 1e-2)
    hi = max(b.max(), m.max())
    bins = np.logspace(np.log10(lo), np.log10(hi), 50)
    axi.hist(b, bins=bins, density=True, color=BENIGN, alpha=0.6,
             label="Benign", edgecolor="white", linewidth=0.2)
    axi.hist(m, bins=bins, density=True, histtype="step", color=MALIG,
             lw=2.5, label="Malignant")
    axi.axvline(b.median(), color=CB["blue"], ls=":", lw=1.5)
    axi.axvline(m.median(), color=MALIG, ls="--", lw=1.5)
    axi.set_xscale("log")
    axi.set_xlabel(label)
    axi.set_ylabel("Density")
    axi.legend(loc="upper right")
    axi.set_title(f"median: benign {b.median():.1f}, malignant {m.median():.1f}")
    _style.panel_label(axi, "ab"[k])
fig.suptitle("Lesion-size distribution by class (log scale)")
paths["fig4_lesion_size.png"] = save_vec(fig, "fig4_lesion_size")

# --- FIG 5: color / border ugly-duckling signals ---
feats5 = [
    ("tbp_lv_H", "Hue angle (tbp_lv_H)"),
    ("tbp_lv_deltaLBnorm", "Lesion-vs-skin contrast (deltaLBnorm)"),
    ("tbp_lv_norm_border", "Border irregularity (norm_border)"),
    ("tbp_lv_norm_color", "Color irregularity (norm_color)"),
    ("tbp_lv_eccentricity", "Eccentricity"),
    ("tbp_lv_radial_color_std_max", "Radial color std (max)"),
]
fig, axes = plt.subplots(2, 3, figsize=(16.5, 9.5), constrained_layout=True)
for k, (axi, (col, label)) in enumerate(zip(axes.ravel(), feats5)):
    b = ben[col].dropna()
    m = mal[col].dropna()
    bins = np.linspace(min(b.min(), m.min()),
                       np.percentile(np.concatenate([b, m]), 99.5), 45)
    axi.hist(b, bins=bins, density=True, color=BENIGN, alpha=0.6,
             label="Benign", edgecolor="white", linewidth=0.2)
    axi.hist(m, bins=bins, density=True, histtype="step", color=MALIG,
             lw=2.5, label="Malignant")
    axi.set_xlabel(label)
    axi.set_ylabel("Density")
    axi.legend()
    _style.panel_label(axi, "abcdef"[k])
fig.suptitle("Dermoscopic colour and border features by class")
paths["fig5_color_border.png"] = save_vec(fig, "fig5_color_border")

# --- FIG 6: correlation heatmap of key numeric TBP features ---
key20 = [
    "clin_size_long_diam_mm", "tbp_lv_areaMM2", "tbp_lv_perimeterMM",
    "tbp_lv_minorAxisMM", "tbp_lv_area_perim_ratio", "tbp_lv_eccentricity",
    "tbp_lv_symm_2axis", "tbp_lv_H", "tbp_lv_A", "tbp_lv_B", "tbp_lv_L",
    "tbp_lv_C", "tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL",
    "tbp_lv_deltaLBnorm", "tbp_lv_color_std_mean", "tbp_lv_radial_color_std_max",
    "tbp_lv_norm_border", "tbp_lv_norm_color", "tbp_lv_nevi_confidence",
]
corr = df[key20].corr().values
labels = [k.replace("tbp_lv_", "").replace("clin_size_long_diam_mm", "clin_diam")
          for k in key20]
fig, ax = plt.subplots(figsize=(12.5, 11), constrained_layout=True)
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=90, fontsize=10)
ax.set_yticklabels(labels, fontsize=10)
for i in range(len(labels)):
    for j in range(len(labels)):
        v = corr[i, j]
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=7.5,
                color="white" if abs(v) > 0.55 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Pearson r")
ax.set_title("Correlation of 21 key TBP geometry and colour features", pad=12)
ax.grid(False)
paths["fig6_correlation.png"] = save_vec(fig, "fig6_correlation")

# --- FIG 0: representative crops by class (quality-filtered) ---
from src.data import HDF5Images

size_mm = dict(zip(df.isic_id, df.clin_size_long_diam_mm))
with HDF5Images(os.path.join(ROOT, "data", "train-image.hdf5")) as imgs:
    def loader(iid):
        return imgs[iid]

    mal_pool = mal.dropna(subset=["clin_size_long_diam_mm", "tbp_lv_norm_color"])
    mal_cand = mal_pool.sort_values("tbp_lv_norm_color", ascending=False)["isic_id"].tolist()
    mal_show = _style.select_clean(loader, mal_cand, 5)

    ben_pool = ben.dropna(subset=["clin_size_long_diam_mm", "tbp_lv_norm_color"])
    ben_pool = ben_pool[ben_pool.tbp_lv_norm_color < ben_pool.tbp_lv_norm_color.median()]
    ben_cand = ben_pool.sample(frac=1.0, random_state=42)["isic_id"].tolist()
    ben_show = _style.select_clean(loader, ben_cand, 5)

    fig, axes = plt.subplots(2, 5, figsize=(14, 6.6))
    for row, ids, col in [(0, mal_show, MALIG), (1, ben_show, CB["green"])]:
        for ax_, iid in zip(axes[row], ids):
            ax_.imshow(loader(iid))
            ax_.set_xticks([]); ax_.set_yticks([]); ax_.grid(False)
            for s in ax_.spines.values():
                s.set_visible(True); s.set_color(col); s.set_linewidth(3)
            ax_.set_title(f"{size_mm.get(iid, float('nan')):.1f} mm", fontsize=12, color=col)
axes[0, 0].set_ylabel("Malignant", fontsize=15, fontweight="bold", color=MALIG)
axes[1, 0].set_ylabel("Benign", fontsize=15, fontweight="bold", color=CB["green"])
fig.suptitle("Representative lesion crops by class", y=0.99)
fig.text(0.5, 0.045,
         "Malignant lesions trend larger, more colour-heterogeneous and more "
         "border-irregular; benign nevi are smaller and more uniform.\n"
         "Each crop is annotated with its clinical longest diameter.",
         ha="center", fontsize=12)
fig.tight_layout(rect=(0, 0.085, 1, 0.95))
paths["fig0_class_examples.png"] = save_png(fig, "fig0_class_examples")

# --- FIG 7: sample crops grid (8 malignant + 8 benign, quality-filtered) ---
with HDF5Images(os.path.join(ROOT, "data", "train-image.hdf5")) as imgs:
    def loader(iid):
        return imgs[iid]

    mal_ids = _style.select_clean(
        loader, mal.sample(frac=1.0, random_state=42).isic_id.tolist(), 8)
    ben_ids = _style.select_clean(
        loader, ben.sample(frac=1.0, random_state=42).isic_id.tolist(), 8)
    grid_ids = mal_ids + ben_ids
    grid_lab = ["MALIGNANT"] * 8 + ["Benign"] * 8
    grid_col = [MALIG] * 8 + [CB["green"]] * 8
    fig, axes = plt.subplots(4, 4, figsize=(12.5, 13.2))
    for ax_, iid, lab, col in zip(axes.ravel(), grid_ids, grid_lab, grid_col):
        ax_.imshow(loader(iid))
        ax_.set_xticks([]); ax_.set_yticks([]); ax_.grid(False)
        for s in ax_.spines.values():
            s.set_visible(True); s.set_color(col); s.set_linewidth(3)
        ax_.set_title(f"{lab}\n{iid}", fontsize=11, color=col, fontweight="bold")
fig.suptitle("Additional sample crops by class (8 malignant, 8 benign)", y=0.995)
fig.tight_layout(rect=(0, 0, 1, 0.965))
paths["fig7_sample_crops.png"] = save_png(fig, "fig7_sample_crops")

# --- FIG 8: image-embedding class separation (PCA-50 -> t-SNE) ---
print("loading embeddings (640-d) ...")
emb_cols = ["isic_id"] + [f"convnextv2_nano_emb{i}" for i in range(640)]
emb = pd.read_parquet(
    os.path.join(ROOT, "experiments", "vision_convnextv2_nano_oof.parquet"),
    columns=emb_cols)
emb = emb.merge(df[["isic_id", "target"]], on="isic_id", how="inner")
pos = emb[emb.target == 1]
neg = emb[emb.target == 0].sample(n=3000, random_state=42)
sub = pd.concat([pos, neg]).reset_index(drop=True)
X = sub[[f"convnextv2_nano_emb{i}" for i in range(640)]].values.astype(np.float32)
ysub = sub.target.values

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

Xs = StandardScaler().fit_transform(X)
Xp = PCA(n_components=50, random_state=42).fit_transform(Xs)
print("running TSNE on", Xp.shape, "...")
Z = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
         random_state=42).fit_transform(Xp)

fig, ax = plt.subplots(figsize=(9.5, 8.5), constrained_layout=True)
ax.scatter(Z[ysub == 0, 0], Z[ysub == 0, 1], s=10, c=BENIGN, alpha=0.45,
           label=f"Benign (n={int((ysub == 0).sum()):,} sampled)", edgecolors="none")
ax.scatter(Z[ysub == 1, 0], Z[ysub == 1, 1], s=46, c=MALIG, alpha=0.9,
           label=f"Malignant (n={int((ysub == 1).sum())}, all)",
           edgecolors="black", linewidths=0.5, marker="^")
ax.set_xlabel("t-SNE dimension 1")
ax.set_ylabel("t-SNE dimension 2")
ax.set_title("ConvNeXtV2-nano OOF embeddings (PCA-50 $\\rightarrow$ t-SNE)")
ax.legend(loc="best")
ax.grid(alpha=0.15)
paths["fig8_embedding_tsne.png"] = save_vec(fig, "fig8_embedding_tsne")

# --- FIG 9: ugly-duckling illustration (within-patient outlier) ---
rng = np.random.RandomState(42)
ud_patients = ["IP_4332092", "IP_0273153", "IP_3987348"]
udfeat = "clin_size_long_diam_mm"
udlabel = "Clinical longest diameter (mm)"
fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.6), constrained_layout=True)
for k, (ax_, pid) in enumerate(zip(axes, ud_patients)):
    sub = df[df.patient_id == pid]
    bvals = sub[sub.target == 0][udfeat].dropna().values
    mvals = sub[sub.target == 1][udfeat].dropna().values
    jit = rng.uniform(-0.12, 0.12, len(bvals))
    ax_.scatter(jit, bvals, s=30, c=BENIGN, alpha=0.7, label="benign lesions",
                edgecolors="none")
    pct = (bvals < mvals[0]).mean() * 100
    ax_.scatter([0], mvals, s=260, marker="*", c=MALIG, edgecolors="black",
                linewidths=1.2, zorder=5, label="malignant")
    ax_.axhline(np.median(bvals), color=CB["grey"], ls="--", lw=1,
                label="patient benign median")
    top = max(bvals.max(), mvals.max())
    ax_.set_ylim(top=top * 1.18)
    ax_.set_xlim(-0.6, 0.6)
    ax_.set_xticks([])
    ax_.set_ylabel(udlabel)
    ax_.set_title(f"{pid}\nn={len(sub)} lesions, malignant at {pct:.0f}th pct")
    ax_.legend(loc="upper left")
    _style.panel_label(ax_, "abc"[k])
fig.suptitle("Within-patient size deviation of the malignant lesion (ugly-duckling sign)")
paths["fig9_ugly_duckling.png"] = save_vec(fig, "fig9_ugly_duckling")

# --- Stats for EDA.md ---
stats = {}
stats["N"] = N
stats["NPOS"] = NPOS
stats["NNEG"] = N - NPOS
stats["PREV"] = PREV
stats["NPAT"] = NPAT
stats["ncols"] = df.shape[1]
stats["dtypes"] = df.dtypes.astype(str).value_counts().to_dict()
stats["median_lpp"] = float(pp.n.median())
stats["mean_lpp"] = float(pp.n.mean())
stats["max_lpp"] = int(pp.n.max())
stats["n_mal_pat"] = n_mal_pat
stats["mal_pat_share"] = float(100 * mal_pat.n.sum() / N)
stats["site_rate"] = st["mean"].to_dict()
stats["site_count"] = st["count"].to_dict()
stats["sex"] = df.groupby("sex")["target"].agg(["count", "mean"]).to_dict("index")
stats["age_mal_med"] = float(mal.age_approx.median())
stats["age_ben_med"] = float(ben.age_approx.median())
stats["paths"] = paths

leak = ["iddx_full", "iddx_1", "iddx_2", "iddx_3", "iddx_4", "iddx_5",
        "mel_mitotic_index", "mel_thick_mm", "lesion_id", "tbp_lv_dnn_lesion_confidence"]
meta_nonpred = ["isic_id", "target", "patient_id", "image_type", "attribution",
                "copyright_license", "tbp_tile_type", "tbp_lv_location",
                "tbp_lv_location_simple"]
real_cols = [c for c in df.columns if c not in leak + meta_nonpred]
miss = (df[real_cols].isna().mean() * 100)
stats["missing"] = miss[miss > 0].sort_values(ascending=False).to_dict()
stats["n_real_cols"] = len(real_cols)

mal_in_multi = df[df.target == 1].merge(pp, left_on="patient_id", right_index=True)
stats["mal_in_multilesion_patients"] = int((mal_in_multi.n > 1).sum())


def within_pct(group):
    if group.target.sum() == 0 or len(group) < 5:
        return np.nan
    mv = group.loc[group.target == 1, udfeat]
    bv = group.loc[group.target == 0, udfeat].dropna()
    if len(bv) == 0 or mv.isna().all():
        return np.nan
    return float((bv.values < mv.dropna().values[0]).mean() * 100)


pcts = df.groupby("patient_id").apply(within_pct).dropna()
stats["median_within_pct_size"] = float(pcts.median())
stats["frac_top10pct"] = float((pcts >= 90).mean() * 100)

with open(os.path.join(OUT, "_stats.json"), "w") as f:
    json.dump(stats, f, indent=2, default=str)
print("wrote stats json")
print("DONE figures:", list(paths.keys()))
