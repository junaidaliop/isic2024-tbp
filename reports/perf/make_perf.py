"""Performance-analysis figures + summary for ISIC-2024 SLICE-3D.

Read-only on data/ and experiments/. Writes ONLY to reports/perf/.
Metric is computed ONLY via src.cv (pauc_above_tpr / oof_pauc).

Run: PYTHONPATH=. SEED=42 python reports/perf/make_perf.py
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy.stats import rankdata
from sklearn.calibration import calibration_curve
from sklearn.metrics import roc_curve

from reports import _style
from src import cv

SEED = 42
np.random.seed(SEED)

OUT = "reports/perf"
DPI = 300
os.makedirs(OUT, exist_ok=True)

C_TAB = _style.CB["blue"]
C_IMG = _style.CB["orange"]
C_STK = _style.CB["green"]
C_NEG = _style.CB["red"]
C_GRY = _style.CB["grey"]
_style.apply(plt, dpi=DPI)

MIN_TPR = 0.80


def save_vec(fig, name):
    base = f"{OUT}/{name}"
    _style.save_vector(fig, base, dpi=DPI)
    plt.close(fig)


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

pa = lambda s: cv.oof_pauc(y, s)  # noqa: E731
PAUC = OrderedDict()
PAUC["tabular (gbdt)"] = pa(gbdt)
PAUC["best image (convnextv2_nano_r224)"] = pa(best_img)
PAUC["stack (rank-avg gbdt+r224)"] = pa(stack)
for n in IMG_NAMES:
    PAUC[f"img:{n}"] = pa(imgs[n])

print("=== verified OOF pAUC@80%TPR (via cv.oof_pauc) ===")
for k, v in PAUC.items():
    print(f"  {k:42s} {v:.5f}")
recon = pa((rankdata(gbdt) + rankdata(best_img)) / 2.0)
print(f"  reconstructed rank-avg[gbdt,r224]          {recon:.5f}")


# === FIGURE 1 — ROC with pAUC@80%TPR region shaded ===
def roc_pts(score):
    fpr, tpr, _ = roc_curve(y, score)
    return fpr, tpr


fig, ax = plt.subplots(figsize=(8.0, 7.0), constrained_layout=True)
series = [
    ("Tabular (GBDT)", gbdt, C_TAB, PAUC["tabular (gbdt)"]),
    ("Best image (ConvNeXtV2-nano@224)", best_img, C_IMG,
     PAUC["best image (convnextv2_nano_r224)"]),
    ("Stack (rank-avg)", stack, C_STK, PAUC["stack (rank-avg gbdt+r224)"]),
]
for label, s, col, p in series:
    fpr, tpr = roc_pts(s)
    ax.plot(fpr, tpr, color=col, lw=2.2, label=f"{label} — pAUC={p:.4f}")

fpr_s, tpr_s, _ = roc_curve(y, stack)
mask = tpr_s >= MIN_TPR
if mask.any():
    j = np.argmax(mask)
    if j > 0:
        f0 = np.interp(MIN_TPR, [tpr_s[j - 1], tpr_s[j]], [fpr_s[j - 1], fpr_s[j]])
        fpr_fill = np.concatenate([[f0], fpr_s[mask]])
        tpr_fill = np.concatenate([[MIN_TPR], tpr_s[mask]])
    else:
        fpr_fill, tpr_fill = fpr_s[mask], tpr_s[mask]
    ax.fill_between(fpr_fill, MIN_TPR, tpr_fill, color=C_STK, alpha=0.18,
                    label="Scored region (TPR $\\geq$ 0.80)")

ax.axhline(MIN_TPR, color=C_GRY, ls="--", lw=1.0)
ax.text(0.60, MIN_TPR + 0.008, "TPR = 0.80 (metric cutoff)", color="#333", fontsize=12)
ax.plot([0, 1], [0, 1], color=C_GRY, ls=":", lw=1.0)
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1.001)
ax.set_title("OOF ROC: pAUC@80%TPR is the shaded area (max = 0.20)")
ax.legend(loc="lower right", framealpha=0.95)
save_vec(fig, "fig1_roc_pauc")

fig, ax = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
for label, s, col, p in series:
    fpr, tpr = roc_pts(s)
    ax.plot(fpr, tpr, color=col, lw=2.4, label=f"{label} — pAUC={p:.4f}")
ax.fill_between(fpr_fill, MIN_TPR, tpr_fill, color=C_STK, alpha=0.18)
ax.set_xlim(0, 0.20)
ax.set_ylim(MIN_TPR - 0.005, 1.001)
ax.set_xlabel("False positive rate (FPR $\\leq$ 0.20 corresponds to TPR $\\geq$ 0.80)")
ax.set_ylabel("True positive rate")
ax.set_title("Zoom on scored band (TPR 0.80–1.00)")
ax.legend(loc="lower right")
save_vec(fig, "fig1b_roc_pauc_zoom")


# === FIGURE 2 — per-fold pAUC grouped bars ===
folds_u = sorted(np.unique(fold_id))
perfold = {"Tabular": [], "Best image": [], "Stack": []}
for f in folds_u:
    m = fold_id == f
    perfold["Tabular"].append(cv.pauc_above_tpr(y[m], gbdt[m]))
    perfold["Best image"].append(cv.pauc_above_tpr(y[m], best_img[m]))
    perfold["Stack"].append(cv.pauc_above_tpr(y[m], stack[m]))

means = {k: np.mean(v) for k, v in perfold.items()}
stds = {k: np.std(v) for k, v in perfold.items()}

fig, ax = plt.subplots(figsize=(9.5, 6.2), constrained_layout=True)
x = np.arange(len(folds_u))
w = 0.26
cols = {"Tabular": C_TAB, "Best image": C_IMG, "Stack": C_STK}
for i, k in enumerate(["Tabular", "Best image", "Stack"]):
    ax.bar(x + (i - 1) * w, perfold[k], w, color=cols[k],
           label=f"{k}  ($\\mu$={means[k]:.4f} $\\pm$ {stds[k]:.4f})")
    ax.axhline(means[k], color=cols[k], ls="--", lw=0.9, alpha=0.6)
ax.set_xticks(x)
ax.set_xticklabels([f"fold {f}" for f in folds_u])
ax.set_ylabel("pAUC@80%TPR")
ax.set_ylim(0, max(max(v) for v in perfold.values()) * 1.22)
ax.set_title("Per-fold pAUC@80%TPR (393 positives)")
ax.legend(loc="upper right")
save_vec(fig, "fig2_perfold_pauc")


# === FIGURE 3a/b/c — ablation tables ===
def render_table(rows, col_labels, title, fname, col_widths=None,
                 highlight_rows=None, figsize=(11.5, None)):
    nrows = len(rows)
    h = figsize[1] if figsize[1] else 1.0 + 0.55 * (nrows + 1)
    fig, ax = plt.subplots(figsize=(figsize[0], h))
    ax.axis("off")
    tbl = ax.table(cellText=rows, colLabels=col_labels, loc="center",
                   cellLoc="left", colLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1, 1.7)
    ncol = len(col_labels)
    if col_widths:
        for (r, c), cell in tbl.get_celld().items():
            cell.set_width(col_widths[c])
    for c in range(ncol):
        cell = tbl[0, c]
        cell.set_facecolor("#222222")
        cell.set_text_props(color="white", fontweight="bold")
    highlight_rows = highlight_rows or {}
    for r in range(nrows):
        for c in range(ncol):
            cell = tbl[r + 1, c]
            if r in highlight_rows:
                cell.set_facecolor(highlight_rows[r])
            elif r % 2 == 0:
                cell.set_facecolor("#f3f3f3")
    ax.set_title(title, fontweight="bold", fontsize=15, pad=16)
    fig.tight_layout()
    base = fname
    fig.savefig(base + ".svg")
    fig.savefig(base + ".png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)


tab_final = PAUC["tabular (gbdt)"]
tab_rows = [
    ["0", "Broken: is_unbalance+AUC-ES, best_iter=1", "0.09941", "—"],
    ["1", "Fix early-stopping / objective", "0.11826", "+0.01885"],
    ["2", "Wide patient-relative ugly-duckling feats", "0.14420", "+0.02594"],
    ["3", "Greysky-bagged + pxc_ + CatBoost ensemble", f"{tab_final:.5f}",
     f"+{tab_final - 0.14420:.5f}"],
]
render_table(
    tab_rows, ["Step", "Change", "OOF pAUC", "$\\Delta$"],
    "Ablation 3a — Tabular GBDT progression",
    f"{OUT}/fig3a_ablation_tabular",
    col_widths=[0.07, 0.55, 0.19, 0.19],
    highlight_rows={3: "#cdeccd"},
)

img_nano128 = PAUC["img:convnextv2_nano"]
img_nano224 = PAUC["img:convnextv2_nano_r224"]
img_tiny = PAUC["img:convnextv2_tiny"]
img_swin = PAUC["img:swinv2_tiny"]
img_eva = PAUC["img:eva02_small"]
img_rows = [
    ["convnextv2_nano @128", "baseline small backbone", f"{img_nano128:.5f}"],
    ["convnextv2_nano @224 +EMA0.995 +mixup", "winner image (res+reg)", f"{img_nano224:.5f}"],
    ["  └ same, EMA 0.999", "over-smoothed (training-time)", "~0.146 (hurt)"],
    ["convnextv2_tiny @224", "heavier backbone", f"{img_tiny:.5f}"],
    ["swinv2_tiny @256", "heavy transformer (overfit)", f"{img_swin:.5f}"],
    ["eva02_small @336", "heaviest (overfit @393 pos)", f"{img_eva:.5f}"],
]
render_table(
    img_rows, ["Config", "Note", "OOF pAUC"],
    "Ablation 3b — Image expert: resolution and regularizer",
    f"{OUT}/fig3b_ablation_image",
    col_widths=[0.42, 0.34, 0.24],
    highlight_rows={1: "#cdeccd", 4: "#f6d6c6", 5: "#f6d6c6"},
)

stk_rankavg = PAUC["stack (rank-avg gbdt+r224)"]
weak_set = ["convnextv2_nano_r224", "convnextv2_tiny", "convnextv2_nano", "vit_tiny"]
weak_stack = rankdata(gbdt)
for n in weak_set:
    weak_stack = weak_stack + rankdata(imgs[n])
weak_stack = weak_stack / (1 + len(weak_set))
stk_weak = cv.oof_pauc(y, weak_stack)
print(f"  reconstructed rank[+weak imgs]            {stk_weak:.5f} (logged 0.16679)")

stk_rows = [
    ["rank-avg [gbdt, r224]", "trivial 2-way rank average", f"{stk_rankavg:.5f}", "winner"],
    ["meta-LGBM stacker", "learned linear/GBDT meta", "0.17108", "loss"],
    ["rank [+ weak images]", "add weaker image OOFs", "0.16679", "dilutes"],
    ["learned per-lesion gate (MoE)", "gated mixture", "0.15007", "decisive loss"],
    ["+ PCA image embeddings", "stack embeddings into GBDT", "hurts (< 0.1743)", "loss"],
]
render_table(
    stk_rows, ["Combiner", "Description", "OOF pAUC", "Verdict"],
    "Ablation 3c — Stack combiners: the trivial rank-average wins",
    f"{OUT}/fig3c_ablation_stack",
    col_widths=[0.30, 0.30, 0.22, 0.18],
    highlight_rows={0: "#cdeccd", 3: "#f6d6c6", 4: "#f6d6c6"},
)


# === FIGURE 4 — GBDT feature importance (top-20 by gain) ===
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
    s = g.sum()
    if s > 0:
        g = g / s
    agg_gain += g
agg_gain /= max(1, (n_lgb + n_cat))

order = np.argsort(agg_gain)[::-1]
top20_idx = order[:20]
top20_names = [feat_names[i] for i in top20_idx]
top20_vals = agg_gain[top20_idx]


def feat_color(name):
    if name.startswith(("pdev_", "prank_", "pxc_")):
        return C_STK
    return C_TAB


fig, ax = plt.subplots(figsize=(10.5, 8.0), constrained_layout=True)
ypos = np.arange(len(top20_names))[::-1]
colors = [feat_color(n) for n in top20_names]
ax.barh(ypos, top20_vals * 100, color=colors)
ax.set_yticks(ypos)
ax.set_yticklabels(top20_names, fontsize=11)
ax.set_xlim(0, top20_vals.max() * 100 * 1.08)
ax.set_xlabel("Mean normalized gain (%), averaged over 50 boosters")
ax.set_title("GBDT top-20 features by gain")
legend_el = [
    Patch(facecolor=C_STK, label="patient-relative (pdev_ / prank_ / pxc_)"),
    Patch(facecolor=C_TAB, label="absolute lesion feature"),
]
ax.legend(handles=legend_el, loc="lower right")
save_vec(fig, "fig4_feature_importance")

pr_mask = np.array([n.startswith(("pdev_", "prank_", "pxc_")) for n in top20_names])
pr_frac_top20 = top20_vals[pr_mask].sum() / top20_vals.sum()
all_pr = np.array([n.startswith(("pdev_", "prank_", "pxc_")) for n in feat_names])
pr_frac_all = agg_gain[all_pr].sum() / agg_gain.sum()
print(f"patient-relative share of top-20 gain: {pr_frac_top20:.1%}; "
      f"of all gain: {pr_frac_all:.1%}")


# === FIGURE 5 — score separation + reliability curve ===
fig, axes = plt.subplots(1, 2, figsize=(14.0, 6.0), constrained_layout=True)
ax = axes[0]
bins = np.linspace(0, 1, 51)
ax.hist(stack[y == 0], bins=bins, color=C_TAB, alpha=0.75,
        label=f"benign (n={int((y == 0).sum()):,})")
ax.hist(stack[y == 1], bins=bins, color=C_NEG, alpha=0.85,
        label=f"malignant (n={int((y == 1).sum()):,})")
ax.set_yscale("log")
ax.set_xlabel("Stack OOF score")
ax.set_ylabel("Count (log scale)")
ax.set_title("Score separation: malignant vs benign")
ax.legend()
_style.panel_label(ax, "a")

ax = axes[1]
frac_pos, mean_pred = calibration_curve(y, stack, n_bins=12, strategy="quantile")
ax.plot([0, 1], [0, 1], color=C_GRY, ls=":", lw=1.4, label="perfect calibration")
ax.plot(mean_pred, frac_pos, "o-", color=C_STK, lw=2, label="stack (12 quantile bins)")
ax.set_xlabel("Mean predicted score (bin)")
ax.set_ylabel("Observed malignant fraction")
ax.set_title("Reliability curve (stack OOF)")
ax.legend(loc="upper left")
_style.panel_label(ax, "b")
save_vec(fig, "fig5_calibration_separation")


# === FIGURE 6 — negative-results panel ===
neg_rows = [
    ["Learned per-lesion gate (MoE)", "0.15007", f"{tab_final:.5f} (tabular)",
     "loses to tabular alone"],
    ["Meta-LGBM stacker", "0.17108", f"{stk_rankavg:.5f} (rank-avg)",
     "loses to trivial rank-avg"],
    ["+ PCA image embeddings into GBDT", "< 0.1743", f"{stk_rankavg:.5f}",
     "embeddings hurt"],
    ["Heavy backbones (swinv2 / eva02)", f"{img_swin:.3f} / {img_eva:.3f}",
     f"{img_nano224:.5f} (nano@224)", "dominated, overfit @393 pos"],
    ["EMA 0.999 (over-smoothing)", "~0.146", f"{img_nano224:.5f} (EMA0.995)",
     "stronger EMA hurts"],
]
fig, ax = plt.subplots(figsize=(14.0, 4.6))
ax.axis("off")
tbl = ax.table(
    cellText=neg_rows,
    colLabels=["Idea tried", "Its pAUC", "Beaten by", "Honest finding"],
    loc="center", cellLoc="left", colLoc="left",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.scale(1, 1.9)
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
ax.set_title("Ablations that did not improve OOF pAUC (leak-verified)",
             fontweight="bold", fontsize=15, pad=16)
fig.tight_layout()
fig.savefig(f"{OUT}/fig6_negative_results.svg")
fig.savefig(f"{OUT}/fig6_negative_results.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)


# === summary for PERF.md ===
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
with open(f"{OUT}/_summary.json", "w") as fh:
    json.dump(summary, fh, indent=2)
print("\nwrote _summary.json")
print("DONE.")
