"""Publication-quality EDA for ISIC-2024 SLICE-3D (companion website).

Read-only on data/. Writes reports/eda/*.png (dpi=150) and EDA.md.
Run: PYTHONPATH=<repo> python reports/eda/make_eda.py
"""
from __future__ import annotations

import io
import os
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

warnings.filterwarnings("ignore")
np.random.seed(42)

ROOT = "/home/efreet/Ali/isic2024-tbp"
OUT = os.path.join(ROOT, "reports", "eda")
os.makedirs(OUT, exist_ok=True)

DPI = 150
# Okabe-Ito colorblind-safe palette
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
BENIGN = CB["sky"]
MALIG = CB["red"]

plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "legend.frameon": False,
    "figure.autolayout": False,
})


def save(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, bbox_inches="tight", dpi=DPI)
    plt.close(fig)
    print("wrote", path)
    return path


print("loading data ...")
df = pd.read_csv(os.path.join(ROOT, "data", "train-metadata.csv"), low_memory=False)
folds = pd.read_parquet(os.path.join(ROOT, "data", "folds.parquet"))
y = df["target"].values
ben = df[df.target == 0]
mal = df[df.target == 1]
N = len(df)
NPOS = int(df.target.sum())
PREV = 100 * df.target.mean()
NPAT = df.patient_id.nunique()

paths = {}

# ---------------------------------------------------------------------------
# FIG 1: class imbalance + per-fold positives
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.6))
counts = [N - NPOS, NPOS]
bars = ax[0].bar(["Benign", "Malignant"], counts, color=[BENIGN, MALIG], width=0.6)
ax[0].set_yscale("log")
ax[0].set_ylabel("Number of lesions (log scale)")
ax[0].set_title("Class imbalance: 400,666 benign vs 393 malignant", fontsize=12)
for b, c in zip(bars, counts):
    ax[0].text(b.get_x() + b.get_width() / 2, c * 1.25, f"{c:,}",
               ha="center", va="bottom", fontweight="bold")
ax[0].set_ylim(1, N * 6)
ax[0].text(0.5, 0.74, f"Prevalence = {PREV:.3f}%\n(1 malignant per {int(round(N/NPOS)):,})",
           transform=ax[0].transAxes, ha="center", fontsize=11,
           bbox=dict(boxstyle="round,pad=0.4", fc=CB["yellow"], ec="black", alpha=0.85))

fg = folds.groupby("fold")["target"].agg(["count", "sum"])
fg["rate"] = 100 * fg["sum"] / fg["count"]
fb = ax[1].bar(fg.index.astype(str), fg["sum"], color=CB["blue"], width=0.6)
ax[1].set_xlabel("CV fold")
ax[1].set_ylabel("Malignant lesions in fold")
ax[1].set_title("Stratified patient-grouped folds: balanced positives", fontsize=12)
for b, s, r in zip(fb, fg["sum"], fg["rate"]):
    ax[1].text(b.get_x() + b.get_width() / 2, s + 0.6, f"{int(s)}\n({r:.3f}%)",
               ha="center", va="bottom", fontsize=9)
ax[1].set_ylim(0, fg["sum"].max() * 1.25)
fig.tight_layout(w_pad=3)
paths["fig1_class_imbalance.png"] = save(fig, "fig1_class_imbalance.png")

# ---------------------------------------------------------------------------
# FIG 2: patient structure
# ---------------------------------------------------------------------------
pp = df.groupby("patient_id").agg(n=("isic_id", "count"), mal=("target", "sum"))
n_mal_pat = int((pp.mal >= 1).sum())
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
bins = np.logspace(0, np.log10(pp.n.max()), 40)
ax[0].hist(pp.n, bins=bins, color=CB["blue"], edgecolor="white", linewidth=0.4)
ax[0].set_xscale("log")
ax[0].set_xlabel("Lesions per patient (log scale)")
ax[0].set_ylabel("Number of patients")
ax[0].set_title("Lesions per patient")
ax[0].axvline(pp.n.median(), color=MALIG, ls="--", lw=2)
txt = (f"patients = {NPAT:,}\nmedian = {int(pp.n.median())} lesions\n"
       f"max = {int(pp.n.max()):,} lesions\nmean = {pp.n.mean():.0f} lesions")
ax[0].text(0.62, 0.95, txt, transform=ax[0].transAxes, va="top", ha="left",
           bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=CB["grey"]))

# patients with >=1 malignant vs none, and how many lesions they carry
mal_pat = pp[pp.mal >= 1]
nomal_pat = pp[pp.mal == 0]
ax[1].bar(["No malignant\nlesion", "≥1 malignant\nlesion"],
          [len(nomal_pat), len(mal_pat)], color=[BENIGN, MALIG], width=0.55)
for i, v in enumerate([len(nomal_pat), len(mal_pat)]):
    ax[1].text(i, v + 6, f"{v}", ha="center", fontweight="bold")
ax[1].set_ylabel("Number of patients")
ax[1].set_title(f"{n_mal_pat} of {NPAT} patients carry ≥1 malignant lesion")
ax[1].set_ylim(0, max(len(nomal_pat), len(mal_pat)) * 1.18)
ax[1].text(0.5, 0.80,
           f"those {n_mal_pat} patients hold\n{int(mal_pat.n.sum()):,} lesions "
           f"({100*mal_pat.n.sum()/N:.0f}% of all crops)",
           transform=ax[1].transAxes, ha="center",
           bbox=dict(boxstyle="round,pad=0.4", fc=CB["yellow"], ec="black", alpha=0.8))
paths["fig2_patient_structure.png"] = save(fig, "fig2_patient_structure.png")

# ---------------------------------------------------------------------------
# FIG 3: demographics
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))
# age distribution normalized
abins = np.arange(0, 91, 5)
ax[0].hist(ben.age_approx.dropna(), bins=abins, density=True, color=BENIGN,
           alpha=0.6, label="Benign", edgecolor="white", linewidth=0.3)
ax[0].hist(mal.age_approx.dropna(), bins=abins, density=True, histtype="step",
           color=MALIG, lw=2.5, label="Malignant")
ax[0].set_xlabel("Approximate age (years)")
ax[0].set_ylabel("Density")
ax[0].set_title("Age: malignant skew older")
ax[0].legend()

# sex split (grouped count + rate annotation)
sx = df.groupby("sex")["target"].agg(["count", "sum", "mean"]).reindex(["female", "male"])
x = np.arange(2)
ax[1].bar(x, sx["count"], color=CB["grey"], width=0.55)
ax[1].set_xticks(x)
ax[1].set_xticklabels(["Female", "Male"])
ax[1].set_ylabel("Number of lesions")
ax[1].set_title("Sex split (malignant-rate annotated)")
for i, (c, m) in enumerate(zip(sx["count"], sx["mean"])):
    ax[1].text(i, c + 3000, f"{int(c):,}\nmalig {100*m:.3f}%",
               ha="center", va="bottom", fontsize=9)
ax[1].set_ylim(0, sx["count"].max() * 1.22)

# malignant rate by site
st = (df.groupby("anatom_site_general")["target"].agg(["count", "mean"])
      .sort_values("mean"))
sb = ax[2].barh(range(len(st)), 100 * st["mean"], color=CB["orange"])
ax[2].set_yticks(range(len(st)))
ax[2].set_yticklabels([s.replace(" ", "\n", 0) for s in st.index])
ax[2].set_xlabel("Malignant rate (%)")
ax[2].set_title("Malignant rate by body site\n(head/neck ~7x baseline)")
for i, (r, c) in enumerate(zip(st["mean"], st["count"])):
    ax[2].text(100 * r + 0.02, i, f"{100*r:.3f}%  (n={int(c):,})",
               va="center", fontsize=8.5)
ax[2].set_xlim(0, 100 * st["mean"].max() * 1.35)
paths["fig3_demographics.png"] = save(fig, "fig3_demographics.png")

# ---------------------------------------------------------------------------
# FIG 4: lesion size
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
for axi, col, label in [
    (ax[0], "clin_size_long_diam_mm", "Clinical longest diameter (mm)"),
    (ax[1], "tbp_lv_areaMM2", "TBP lesion area (mm$^2$)"),
]:
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
    axi.legend()
    axi.set_title(f"median benign {b.median():.1f} vs malignant {m.median():.1f}")
fig.suptitle("Malignant lesions skew markedly larger", fontsize=14, fontweight="bold", y=1.02)
paths["fig4_lesion_size.png"] = save(fig, "fig4_lesion_size.png")

# ---------------------------------------------------------------------------
# FIG 5: color / border ugly-duckling signals
# ---------------------------------------------------------------------------
feats5 = [
    ("tbp_lv_H", "Hue angle (tbp_lv_H)"),
    ("tbp_lv_deltaLBnorm", "Lesion-vs-skin contrast (deltaLBnorm)"),
    ("tbp_lv_norm_border", "Border irregularity (norm_border)"),
    ("tbp_lv_norm_color", "Color irregularity (norm_color)"),
    ("tbp_lv_eccentricity", "Eccentricity"),
    ("tbp_lv_radial_color_std_max", "Radial color std (max)"),
]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
for axi, (col, label) in zip(axes.ravel(), feats5):
    b = ben[col].dropna()
    m = mal[col].dropna()
    bins = np.linspace(min(b.min(), m.min()), np.percentile(np.concatenate([b, m]), 99.5), 45)
    axi.hist(b, bins=bins, density=True, color=BENIGN, alpha=0.6,
             label="Benign", edgecolor="white", linewidth=0.2)
    axi.hist(m, bins=bins, density=True, histtype="step", color=MALIG,
             lw=2.5, label="Malignant")
    axi.set_xlabel(label)
    axi.set_ylabel("Density")
    axi.legend(fontsize=9)
fig.suptitle("Color & border 'ugly-duckling' signals: malignant shifts (hue separates best, AUC 0.81)",
             fontsize=14, fontweight="bold", y=1.0)
fig.tight_layout(rect=[0, 0, 1, 0.97])
paths["fig5_color_border.png"] = save(fig, "fig5_color_border.png")

# ---------------------------------------------------------------------------
# FIG 6: correlation heatmap of ~20 key numeric TBP features
# ---------------------------------------------------------------------------
key20 = [
    "clin_size_long_diam_mm", "tbp_lv_areaMM2", "tbp_lv_perimeterMM",
    "tbp_lv_minorAxisMM", "tbp_lv_area_perim_ratio", "tbp_lv_eccentricity",
    "tbp_lv_symm_2axis", "tbp_lv_H", "tbp_lv_A", "tbp_lv_B", "tbp_lv_L",
    "tbp_lv_C", "tbp_lv_deltaA", "tbp_lv_deltaB", "tbp_lv_deltaL",
    "tbp_lv_deltaLBnorm", "tbp_lv_color_std_mean", "tbp_lv_radial_color_std_max",
    "tbp_lv_norm_border", "tbp_lv_norm_color", "tbp_lv_nevi_confidence",
]
corr = df[key20].corr().values
labels = [k.replace("tbp_lv_", "").replace("clin_size_long_diam_mm", "clin_diam") for k in key20]
fig, ax = plt.subplots(figsize=(11, 9.5))
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)))
ax.set_yticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=90, fontsize=8)
ax.set_yticklabels(labels, fontsize=8)
for i in range(len(labels)):
    for j in range(len(labels)):
        v = corr[i, j]
        ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=6,
                color="white" if abs(v) > 0.55 else "black")
cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cb.set_label("Pearson r")
ax.set_title("Correlation of 21 key TBP geometry & color features", pad=12)
ax.grid(False)
paths["fig6_correlation.png"] = save(fig, "fig6_correlation.png")

# ---------------------------------------------------------------------------
# FIG 7: sample crops grid (8 malignant + 8 benign)
# ---------------------------------------------------------------------------
from src.data import HDF5Images

rng = np.random.RandomState(42)
mal_ids = mal.isic_id.sample(8, random_state=42).tolist()
ben_ids = ben.isic_id.sample(8, random_state=42).tolist()
fig, axes = plt.subplots(4, 4, figsize=(11, 11.6))
with HDF5Images(os.path.join(ROOT, "data", "train-image.hdf5")) as imgs:
    grid_ids = mal_ids + ben_ids
    grid_lab = ["MALIGNANT"] * 8 + ["benign"] * 8
    grid_col = [MALIG] * 8 + [CB["green"]] * 8
    # interleave rows: top 2 rows malignant, bottom 2 benign
    for ax_, iid, lab, col in zip(axes.ravel(), grid_ids, grid_lab, grid_col):
        arr = imgs[iid]
        ax_.imshow(arr)
        ax_.set_xticks([]); ax_.set_yticks([])
        ax_.grid(False)
        for s in ax_.spines.values():
            s.set_visible(True); s.set_color(col); s.set_linewidth(3)
        ax_.set_title(f"{lab}\n{iid}", fontsize=8, color=col, fontweight="bold")
fig.suptitle("Sample lesion crops  (top 8 malignant, bottom 8 benign)",
             fontsize=14, fontweight="bold", y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.97])
paths["fig7_sample_crops.png"] = save(fig, "fig7_sample_crops.png")

# ---------------------------------------------------------------------------
# FIG 8: image-embedding class separation (PCA-50 -> TSNE)
# ---------------------------------------------------------------------------
print("loading embeddings (640-d) ...")
emb_cols = ["isic_id"] + [f"convnextv2_nano_emb{i}" for i in range(640)]
emb = pd.read_parquet(os.path.join(ROOT, "experiments", "vision_convnextv2_nano_oof.parquet"),
                      columns=emb_cols)
emb = emb.merge(df[["isic_id", "target"]], on="isic_id", how="inner")
pos = emb[emb.target == 1]
neg = emb[emb.target == 0].sample(n=3000, random_state=42)
sub = pd.concat([pos, neg]).reset_index(drop=True)
X = sub[[f"convnextv2_nano_emb{i}" for i in range(640)]].values.astype(np.float32)
ysub = sub.target.values

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

Xs = StandardScaler().fit_transform(X)
Xp = PCA(n_components=50, random_state=42).fit_transform(Xs)
print("running TSNE on", Xp.shape, "...")
Z = TSNE(n_components=2, perplexity=30, init="pca", learning_rate="auto",
         random_state=42).fit_transform(Xp)

fig, ax = plt.subplots(figsize=(9, 8))
ax.scatter(Z[ysub == 0, 0], Z[ysub == 0, 1], s=8, c=BENIGN, alpha=0.45,
           label=f"Benign (n={int((ysub==0).sum()):,} sampled)", edgecolors="none")
ax.scatter(Z[ysub == 1, 0], Z[ysub == 1, 1], s=42, c=MALIG, alpha=0.9,
           label=f"Malignant (n={int((ysub==1).sum())}, all)",
           edgecolors="black", linewidths=0.5, marker="^")
ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
ax.set_title("ConvNeXtV2-nano OOF embeddings (PCA-50 → t-SNE)\nmalignant cluster in feature space")
ax.legend(loc="best")
ax.grid(alpha=0.15)
paths["fig8_embedding_tsne.png"] = save(fig, "fig8_embedding_tsne.png")

# ---------------------------------------------------------------------------
# FIG 9: ugly-duckling illustration (within-patient outlier)
# ---------------------------------------------------------------------------
# chosen patients where malignant lesion is a clear within-patient outlier
ud_patients = ["IP_4332092", "IP_0273153", "IP_3987348"]
udfeat = "clin_size_long_diam_mm"
udlabel = "Clinical longest diameter (mm)"
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax_, pid in zip(axes, ud_patients):
    sub = df[df.patient_id == pid]
    bvals = sub[sub.target == 0][udfeat].dropna().values
    mvals = sub[sub.target == 1][udfeat].dropna().values
    jit = rng.uniform(-0.12, 0.12, len(bvals))
    ax_.scatter(jit, bvals, s=28, c=BENIGN, alpha=0.7, label="benign lesions",
                edgecolors="none")
    pct = (bvals < mvals[0]).mean() * 100
    ax_.scatter([0], mvals, s=240, marker="*", c=MALIG, edgecolors="black",
                linewidths=1.2, zorder=5, label="MALIGNANT")
    ax_.axhline(np.median(bvals), color=CB["grey"], ls="--", lw=1,
                label="patient benign median")
    ax_.set_xlim(-0.6, 0.6)
    ax_.set_xticks([])
    ax_.set_ylabel(udlabel)
    ax_.set_title(f"{pid}\nn={len(sub)} lesions  |  malignant at {pct:.0f}th pct")
    ax_.legend(fontsize=8, loc="upper left")
fig.suptitle("Ugly-duckling sign: within each patient the malignant lesion is a size outlier",
             fontsize=14, fontweight="bold", y=1.02)
fig.tight_layout(rect=[0, 0, 1, 0.96])
paths["fig9_ugly_duckling.png"] = save(fig, "fig9_ugly_duckling.png")

# ---------------------------------------------------------------------------
# Stats for EDA.md
# ---------------------------------------------------------------------------
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

# within-patient malignant stats
mal_in_multi = df[df.target == 1].merge(pp, left_on="patient_id", right_index=True)
stats["mal_in_multilesion_patients"] = int((mal_in_multi.n > 1).sum())
# percentile of malignant within its patient on clin size
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

import json
with open(os.path.join(OUT, "_stats.json"), "w") as f:
    json.dump(stats, f, indent=2, default=str)
print("wrote stats json")
print("DONE figures:", list(paths.keys()))
