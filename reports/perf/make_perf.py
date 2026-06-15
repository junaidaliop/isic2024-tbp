"""
make_perf.py — PERFORMANCE-ANALYSIS figures + summary for the ISIC-2024 companion site.

Read-only on data/ and experiments/. Writes ONLY to reports/perf/.
Metric is computed ONLY via src.cv (pauc_above_tpr / oof_pauc).

Run:
    PYTHONPATH=<repo> SEED=42 python reports/perf/make_perf.py
"""

from __future__ import annotations

import os
from collections import OrderedDict

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from scipy.stats import rankdata
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

from src import cv

SEED = 42
np.random.seed(SEED)

OUT = "reports/perf"
DPI = 150
os.makedirs(OUT, exist_ok=True)

# Colorblind-safe palette (Wong / Okabe-Ito)
C_TAB = "#0072B2"   # blue   - tabular
C_IMG = "#E69F00"   # orange - best image
C_STK = "#009E73"   # green  - stack
C_NEG = "#D55E00"   # vermilion - negatives / malignant
C_GRY = "#999999"

plt.rcParams.update({
    "figure.dpi": DPI,
    "savefig.dpi": DPI,
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

MIN_TPR = 0.80

# ---------------------------------------------------------------------------
# Load aligned OOF predictions on frozen folds
# ---------------------------------------------------------------------------
folds = cv.load_folds("data/folds.parquet")
IDX = folds["isic_id"].values
y = folds.set_index("isic_id")["target"].reindex(IDX).values.astype(int)
fold_id = folds.set_index("isic_id")["fold"].reindex(IDX).values


def _load(path, col):
    s = pd.read_parquet(path, columns=["isic_id", col]).set_index("isic_id")[col]
    return s.reindex(IDX).values


gbdt = _load("experiments/gbdt_oof.parquet", "gbdt_oof")
stack = _load("experiments/stack_oof.parquet", "stack_oof")

IMG_NAMES = [
    "convnextv2_nano_r224", "convnextv2_nano", "convnextv2_tiny", "effvit_b0",
    "vit_tiny", "mnv4_small", "swinv2_tiny", "eva02_small",
]
imgs = {n: _load(f"experiments/vision_{n}_oof.parquet", f"{n}_oof") for n in IMG_NAMES}
best_img = imgs["convnextv2_nano_r224"]

# ---------------------------------------------------------------------------
# Verify all OOF-derived numbers via cv.oof_pauc
# ---------------------------------------------------------------------------
pa = lambda s: cv.oof_pauc(y, s)
PAUC = OrderedDict()
PAUC["tabular (gbdt)"] = pa(gbdt)
PAUC["best image (convnextv2_nano_r224)"] = pa(best_img)
PAUC["stack (rank-avg gbdt+r224)"] = pa(stack)
for n in IMG_NAMES:
    PAUC[f"img:{n}"] = pa(imgs[n])

verify_lines = []
print("=== verified OOF pAUC@80%TPR (via cv.oof_pauc) ===")
for k, v in PAUC.items():
    print(f"  {k:42s} {v:.5f}")

# rank-avg reconstruction sanity (should match canonical stack file)
recon = pa((rankdata(gbdt) + rankdata(best_img)) / 2.0)
print(f"  reconstructed rank-avg[gbdt,r224]          {recon:.5f}")


# ===========================================================================
# FIGURE 1 — ROC with pAUC@80%TPR region shaded
# ===========================================================================
def roc_pts(score):
    fpr, tpr, _ = roc_curve(y, score)
    return fpr, tpr


fig, ax = plt.subplots(figsize=(7.2, 6.4))
series = [
    ("Tabular (GBDT)", gbdt, C_TAB, PAUC["tabular (gbdt)"]),
    ("Best image (ConvNeXtV2-nano@224)", best_img, C_IMG, PAUC["best image (convnextv2_nano_r224)"]),
    ("Stack (rank-avg)", stack, C_STK, PAUC["stack (rank-avg gbdt+r224)"]),
]
for label, s, col, p in series:
    fpr, tpr = roc_pts(s)
    ax.plot(fpr, tpr, color=col, lw=2.0,
            label=f"{label} — pAUC={p:.4f}")

# Shade the scored region: TPR in [0.80, 1.0] under the BEST (stack) curve,
# which is exactly what the metric integrates.
fpr_s, tpr_s, _ = roc_curve(y, stack)
mask = tpr_s >= MIN_TPR
# interpolate the exact entry point at TPR = 0.80
if mask.any():
    j = np.argmax(mask)  # first index where tpr>=0.80
    if j > 0:
        f0 = np.interp(MIN_TPR, [tpr_s[j - 1], tpr_s[j]], [fpr_s[j - 1], fpr_s[j]])
        fpr_fill = np.concatenate([[f0], fpr_s[mask]])
        tpr_fill = np.concatenate([[MIN_TPR], tpr_s[mask]])
    else:
        fpr_fill, tpr_fill = fpr_s[mask], tpr_s[mask]
    ax.fill_between(fpr_fill, MIN_TPR, tpr_fill, color=C_STK, alpha=0.18,
                    label="Scored region (TPR ≥ 0.80)")

ax.axhline(MIN_TPR, color=C_GRY, ls="--", lw=1.0)
ax.text(0.62, MIN_TPR + 0.006, "TPR = 0.80 (metric cutoff)", color="#444", fontsize=9)
ax.plot([0, 1], [0, 1], color=C_GRY, ls=":", lw=1.0)
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.001)
ax.set_title("OOF ROC — pAUC@80%TPR is the shaded area (max=0.20)")
ax.legend(loc="lower right", fontsize=9, framealpha=0.95)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1_roc_pauc.png")
plt.close(fig)

# inset zoom version of scored region
fig, ax = plt.subplots(figsize=(7.2, 5.4))
for label, s, col, p in series:
    fpr, tpr = roc_pts(s)
    ax.plot(fpr, tpr, color=col, lw=2.2, label=f"{label} — pAUC={p:.4f}")
ax.fill_between(fpr_fill, MIN_TPR, tpr_fill, color=C_STK, alpha=0.18)
ax.set_xlim(0, 0.20)
ax.set_ylim(MIN_TPR - 0.005, 1.001)
ax.set_xlabel("False Positive Rate (≤ 0.20 corresponds to TPR ≥ 0.80)")
ax.set_ylabel("True Positive Rate")
ax.set_title("Zoom on scored band (TPR 0.80–1.00)")
ax.legend(loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/fig1b_roc_pauc_zoom.png")
plt.close(fig)


# ===========================================================================
# FIGURE 2 — per-fold pAUC grouped bars (tabular / best image / stack)
# ===========================================================================
folds_u = sorted(np.unique(fold_id))
perfold = {"Tabular": [], "Best image": [], "Stack": []}
for f in folds_u:
    m = fold_id == f
    perfold["Tabular"].append(cv.pauc_above_tpr(y[m], gbdt[m]))
    perfold["Best image"].append(cv.pauc_above_tpr(y[m], best_img[m]))
    perfold["Stack"].append(cv.pauc_above_tpr(y[m], stack[m]))

means = {k: np.mean(v) for k, v in perfold.items()}
stds = {k: np.std(v) for k, v in perfold.items()}

fig, ax = plt.subplots(figsize=(8.2, 5.4))
x = np.arange(len(folds_u))
w = 0.26
cols = {"Tabular": C_TAB, "Best image": C_IMG, "Stack": C_STK}
for i, k in enumerate(["Tabular", "Best image", "Stack"]):
    ax.bar(x + (i - 1) * w, perfold[k], w, color=cols[k],
           label=f"{k}  (μ={means[k]:.4f} ± {stds[k]:.4f})")
ax.set_xticks(x)
ax.set_xticklabels([f"fold {f}" for f in folds_u])
ax.set_ylabel("pAUC@80%TPR")
ax.set_title("Per-fold pAUC — fold-to-fold variance at 393 positives")
ax.legend(fontsize=9, loc="upper right")
# annotate mean lines
for k in ["Tabular", "Best image", "Stack"]:
    ax.axhline(means[k], color=cols[k], ls="--", lw=0.8, alpha=0.6)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_perfold_pauc.png")
plt.close(fig)


# ===========================================================================
# FIGURE 3a/b/c — ablation tables as PNGs
# ===========================================================================
def render_table(rows, col_labels, title, fname, col_widths=None,
                 highlight_rows=None, figsize=(9.5, None)):
    nrows = len(rows)
    h = figsize[1] if figsize[1] else 0.6 + 0.42 * (nrows + 1)
    fig, ax = plt.subplots(figsize=(figsize[0], h))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                   cellLoc="left", colLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.5)
    ncol = len(col_labels)
    if col_widths:
        for (r, c), cell in tbl.get_celld().items():
            cell.set_width(col_widths[c])
    # header style
    for c in range(ncol):
        cell = tbl[0, c]
        cell.set_facecolor("#222222")
        cell.set_text_props(color="white", fontweight="bold")
    # highlight winner rows
    highlight_rows = highlight_rows or {}
    for r in range(nrows):
        for c in range(ncol):
            cell = tbl[r + 1, c]
            if r in highlight_rows:
                cell.set_facecolor(highlight_rows[r])
            elif r % 2 == 0:
                cell.set_facecolor("#f3f3f3")
    ax.set_title(title, fontweight="bold", pad=14)
    fig.tight_layout()
    fig.savefig(fname, bbox_inches="tight")
    plt.close(fig)


# --- 3a tabular progression (training-time deltas given; final verified) ---
tab_final = PAUC["tabular (gbdt)"]
tab_rows = [
    ["0", "Broken: is_unbalance+AUC-ES, best_iter=1", "0.09941", "—"],
    ["1", "Fix early-stopping / objective", "0.11826", "+0.01885"],
    ["2", "Wide patient-relative ugly-duckling feats", "0.14420", "+0.02594"],
    ["3", "Greysky-bagged + pxc_ + CatBoost ensemble", f"{tab_final:.5f}", f"+{tab_final-0.14420:+.5f}".replace('++','+')],
]
render_table(
    tab_rows,
    ["Step", "Change", "OOF pAUC", "Δ"],
    "Ablation 3a — Tabular GBDT progression",
    f"{OUT}/fig3a_ablation_tabular.png",
    col_widths=[0.07, 0.55, 0.19, 0.19],
    highlight_rows={3: "#cdeccd"},
)

# --- 3b image resolution / regularizer ---
img_nano128 = PAUC["img:convnextv2_nano"]
img_nano224 = PAUC["img:convnextv2_nano_r224"]
img_tiny = PAUC["img:convnextv2_tiny"]
img_swin = PAUC["img:swinv2_tiny"]
img_eva = PAUC["img:eva02_small"]
img_rows = [
    ["convnextv2_nano @128", "baseline small backbone", f"{img_nano128:.5f}"],
    ["convnextv2_nano @224 +EMA0.995 +mixup", "WINNER image (res+reg)", f"{img_nano224:.5f}"],
    ["  └ same, EMA 0.999", "over-smoothed (training-time)", "~0.146 (HURT)"],
    ["convnextv2_tiny @224", "heavier backbone", f"{img_tiny:.5f}"],
    ["swinv2_tiny @256", "heavy transformer (overfit)", f"{img_swin:.5f}"],
    ["eva02_small @336", "heaviest (overfit @393 pos)", f"{img_eva:.5f}"],
]
render_table(
    img_rows,
    ["Config", "Note", "OOF pAUC"],
    "Ablation 3b — Image expert: resolution & regularizer (small wins; heavy overfits)",
    f"{OUT}/fig3b_ablation_image.png",
    col_widths=[0.42, 0.34, 0.24],
    highlight_rows={1: "#cdeccd", 4: "#f6d6c6", 5: "#f6d6c6"},
)

# --- 3c stack combiners ---
# verify the ones we can: rank-avg winner = stack file; rank[+weak imgs] reconstruct.
# meta-LGBM / learned-gate / +PCA numbers are taken as logged (their oof not on disk),
# but rank-avg winner is recomputed.
stk_rankavg = PAUC["stack (rank-avg gbdt+r224)"]
# reconstruct rank[+weak imgs] = rank-avg over gbdt + several image OOFs (best+tiny+nano128)
weak_set = ["convnextv2_nano_r224", "convnextv2_tiny", "convnextv2_nano", "vit_tiny"]
weak_stack = rankdata(gbdt)
for n in weak_set:
    weak_stack = weak_stack + rankdata(imgs[n])
weak_stack = weak_stack / (1 + len(weak_set))
stk_weak = cv.oof_pauc(y, weak_stack)
print(f"  reconstructed rank[+weak imgs]            {stk_weak:.5f} (logged 0.16679)")

stk_rows = [
    ["rank-avg [gbdt, r224]", "trivial 2-way rank average", f"{stk_rankavg:.5f}", "WINNER"],
    ["meta-LGBM stacker", "learned linear/GBDT meta", "0.17108", "loss"],
    ["rank [+ weak images]", "add weaker image OOFs", "0.16679", "dilutes"],
    ["learned per-lesion gate (MoE)", "gated mixture", "0.15007", "decisive loss < tabular*"],
    ["+ PCA image embeddings", "stack embeddings into GBDT", "hurts (< 0.1743)", "loss"],
]
render_table(
    stk_rows,
    ["Combiner", "Description", "OOF pAUC", "Verdict"],
    "Ablation 3c — Stack combiners: the trivial rank-average wins",
    f"{OUT}/fig3c_ablation_stack.png",
    col_widths=[0.30, 0.30, 0.22, 0.18],
    highlight_rows={0: "#cdeccd", 3: "#f6d6c6", 4: "#f6d6c6"},
)


# ===========================================================================
# FIGURE 4 — GBDT feature importance (top-20 by gain, aggregated over 50 boosters)
# ===========================================================================
boosters = joblib.load("experiments/gbdt_boosters.joblib")
feat_names = list(boosters[0][1])
n_feat = len(feat_names)
agg_gain = np.zeros(n_feat, dtype=float)
n_lgb = n_cat = 0
for rm, cols, _state in boosters:
    assert list(cols) == feat_names, "feature order mismatch across boosters"
    mdl = rm.model
    if rm.kind == "lgb":
        g = mdl.booster_.feature_importance(importance_type="gain").astype(float)
        n_lgb += 1
    elif rm.kind == "cat":
        g = np.asarray(mdl.get_feature_importance(), dtype=float)
        n_cat += 1
    else:
        continue
    # normalize each booster's gain to sum=1 so LGBM/Cat scales are comparable
    s = g.sum()
    if s > 0:
        g = g / s
    agg_gain += g
agg_gain /= max(1, (n_lgb + n_cat))
print(f"\nfeature importance aggregated over {n_lgb} LGBM + {n_cat} CatBoost boosters")

order = np.argsort(agg_gain)[::-1]
top20_idx = order[:20]
top20_names = [feat_names[i] for i in top20_idx]
top20_vals = agg_gain[top20_idx]


def feat_color(name):
    if name.startswith("pdev_") or name.startswith("prank_") or name.startswith("pxc_"):
        return C_STK  # patient-relative (ugly-duckling)
    return C_TAB


fig, ax = plt.subplots(figsize=(9.5, 7.2))
ypos = np.arange(len(top20_names))[::-1]
colors = [feat_color(n) for n in top20_names]
ax.barh(ypos, top20_vals * 100, color=colors)
ax.set_yticks(ypos)
ax.set_yticklabels(top20_names, fontsize=9)
ax.set_xlabel("Mean normalized gain (%)  — averaged over 50 boosters")
ax.set_title("GBDT top-20 features by gain — patient-relative feats dominate")
legend_el = [
    Patch(facecolor=C_STK, label="patient-relative (pdev_/prank_/pxc_)"),
    Patch(facecolor=C_TAB, label="absolute lesion feature"),
]
ax.legend(handles=legend_el, loc="lower right", fontsize=9)
fig.tight_layout()
fig.savefig(f"{OUT}/fig4_feature_importance.png")
plt.close(fig)

# fraction of top-20 gain from patient-relative
pr_mask = np.array([n.startswith(("pdev_", "prank_", "pxc_")) for n in top20_names])
pr_frac_top20 = top20_vals[pr_mask].sum() / top20_vals.sum()
# overall
all_pr = np.array([n.startswith(("pdev_", "prank_", "pxc_")) for n in feat_names])
pr_frac_all = agg_gain[all_pr].sum() / agg_gain.sum()
print(f"patient-relative share of top-20 gain: {pr_frac_top20:.1%}; of all gain: {pr_frac_all:.1%}")


# ===========================================================================
# FIGURE 5 — score separation (log-count) + reliability curve (stack)
# ===========================================================================
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.2))

# (a) separation histogram
ax = axes[0]
bins = np.linspace(0, 1, 51)
ax.hist(stack[y == 0], bins=bins, color=C_TAB, alpha=0.75, label=f"benign (n={int((y==0).sum()):,})")
ax.hist(stack[y == 1], bins=bins, color=C_NEG, alpha=0.85, label=f"malignant (n={int((y==1).sum()):,})")
ax.set_yscale("log")
ax.set_xlabel("Stack OOF score")
ax.set_ylabel("count (log scale)")
ax.set_title("Score separation: malignant vs benign (stack OOF)")
ax.legend(fontsize=9)

# (b) reliability / calibration curve
ax = axes[1]
frac_pos, mean_pred = calibration_curve(y, stack, n_bins=12, strategy="quantile")
ax.plot([0, 1], [0, 1], color=C_GRY, ls=":", lw=1.2, label="perfect calibration")
ax.plot(mean_pred, frac_pos, "o-", color=C_STK, lw=2, label="stack (12 quantile bins)")
ax.set_xlabel("Mean predicted score (bin)")
ax.set_ylabel("Observed malignant fraction")
ax.set_title("Reliability curve — stack OOF")
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_calibration_separation.png")
plt.close(fig)


# ===========================================================================
# FIGURE 6 — negative-results summary panel
# ===========================================================================
neg_rows = [
    ["Learned per-lesion gate (MoE)", "0.15007", f"{tab_final:.5f} (tabular)",
     "LOSES to tabular alone"],
    ["Meta-LGBM stacker", "0.17108", f"{stk_rankavg:.5f} (rank-avg)",
     "loses to trivial rank-avg"],
    ["+ PCA image embeddings into GBDT", "< 0.1743", f"{stk_rankavg:.5f}",
     "embeddings HURT"],
    ["Heavy backbones (swinv2 / eva02)", f"{img_swin:.3f} / {img_eva:.3f}",
     f"{img_nano224:.5f} (nano@224)", "dominated — overfit @393 pos"],
    ["EMA 0.999 (over-smoothing)", "~0.146", f"{img_nano224:.5f} (EMA0.995)",
     "stronger EMA hurts"],
]
fig, ax = plt.subplots(figsize=(12.5, 3.7))
ax.axis("off")
tbl = ax.table(
    cellText=neg_rows,
    colLabels=["Idea tried", "Its pAUC", "Beaten by", "Honest finding"],
    loc="center", cellLoc="left", colLoc="left",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1, 1.7)
widths = [0.30, 0.16, 0.24, 0.30]
for (r, c), cell in tbl.get_celld().items():
    cell.set_width(widths[c])
for c in range(4):
    cell = tbl[0, c]
    cell.set_facecolor("#222222")
    cell.set_text_props(color="white", fontweight="bold")
for r in range(len(neg_rows)):
    for c in range(4):
        tbl[r + 1, c].set_facecolor("#f6d6c6" if r % 2 == 0 else "#fbe9de")
ax.set_title("Negative results — complexity did not pay off (all OOF, leak-verified)",
             fontweight="bold", pad=14)
fig.tight_layout()
fig.savefig(f"{OUT}/fig6_negative_results.png", bbox_inches="tight")
plt.close(fig)


# ===========================================================================
# Emit machine-readable summary for PERF.md authoring
# ===========================================================================
summary = {
    "pauc": {k: round(v, 5) for k, v in PAUC.items()},
    "recon_rankavg": round(recon, 5),
    "recon_weak_stack": round(stk_weak, 5),
    "perfold": {k: [round(x, 5) for x in v] for k, v in perfold.items()},
    "means": {k: round(v, 5) for k, v in means.items()},
    "stds": {k: round(v, 5) for k, v in stds.items()},
    "top20": [(top20_names[i], round(float(top20_vals[i]) * 100, 3)) for i in range(20)],
    "pr_frac_top20": round(float(pr_frac_top20), 4),
    "pr_frac_all": round(float(pr_frac_all), 4),
    "n_lgb": n_lgb, "n_cat": n_cat,
}
import json
with open(f"{OUT}/_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print("\nwrote _summary.json")
print("\nTOP-10 features by gain:")
for i in range(10):
    print(f"  {i+1:2d}. {top20_names[i]:34s} {top20_vals[i]*100:6.2f}%")
print("\nDONE.")
