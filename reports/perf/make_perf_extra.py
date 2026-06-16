"""Extra performance figures: confusion matrix, PR curve, metrics table, ranked
prediction examples. Computed from the canonical stack OOF.

Run: PYTHONPATH=. python reports/perf/make_perf_extra.py
"""
import io

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from reports import _style
from src import cv

OUT = "reports/perf"
DPI = 300
GREEN = _style.CB["green"]
RED = _style.CB["red"]
_style.apply(plt, dpi=DPI)

folds = pd.read_parquet("data/folds.parquet")[["isic_id", "target"]]
oof = pd.read_parquet("experiments/stack_oof.parquet")
d = oof.merge(folds, on="isic_id")
y = d["target"].to_numpy()
p = d["stack_oof"].to_numpy()

auc = roc_auc_score(y, p)
ap = average_precision_score(y, p)
pauc = cv.pauc_above_tpr(y, p)
fpr, tpr, thr = roc_curve(y, p)


def at_tpr(t):
    i = int(np.searchsorted(tpr, t))
    i = min(i, len(thr) - 1)
    return float(1 - fpr[i]), float(thr[i])


spec80, t80 = at_tpr(0.80)
spec90, _ = at_tpr(0.90)
spec95, _ = at_tpr(0.95)

pred = (p >= t80).astype(int)
TP = int(((pred == 1) & (y == 1)).sum())
FP = int(((pred == 1) & (y == 0)).sum())
TN = int(((pred == 0) & (y == 0)).sum())
FN = int(((pred == 0) & (y == 1)).sum())
sens = TP / (TP + FN)
prec = TP / (TP + FP) if (TP + FP) else 0.0
f1 = 2 * prec * sens / (prec + sens) if (prec + sens) else 0.0

print(f"AUC={auc:.4f}  pAUC@80={pauc:.5f}  AP={ap:.4f}")
print(f"@80%TPR thr={t80:.4f} spec={spec80:.4f} TP={TP} FP={FP} TN={TN} FN={FN} "
      f"prec={prec:.4f} F1={f1:.4f}")

# Confusion at 80%-TPR operating point. Colour by within-class rate so the
# malignant row does not vanish under the 1:1000 ratio.
cm = np.array([[TN, FP], [FN, TP]], dtype=float)
row_rate = cm / cm.sum(axis=1, keepdims=True)
roles = [["TN", "FP"], ["FN", "TP"]]
fig, ax = plt.subplots(figsize=(6.4, 5.8), constrained_layout=True)
im = ax.imshow(row_rate, cmap="Greens", vmin=0.0, vmax=1.0)
for (i, j), r in np.ndenumerate(row_rate):
    ax.text(j, i, f"{roles[i][j]}\n{int(cm[i, j]):,}\n{r:.1%} of row",
            ha="center", va="center", fontsize=14,
            color="white" if r > 0.55 else "#1a1a1a", fontweight="bold")
ax.set_xticks([0, 1], ["predicted benign", "predicted malignant"])
ax.set_yticks([0, 1], ["actual benign", "actual malignant"])
ax.set_title("Confusion matrix at the 80%-TPR operating point\n"
             f"stacked model, OOF · sensitivity {sens:.3f}, specificity {spec80:.3f}",
             fontsize=14)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("within-class rate")
ax.grid(False)
_style.save_vector(fig, f"{OUT}/fig7_confusion_matrix", dpi=DPI)
plt.close(fig)

# Precision-recall curve
pr, rc, _ = precision_recall_curve(y, p)
fig, ax = plt.subplots(figsize=(6.6, 5.6), constrained_layout=True)
ax.plot(rc, pr, color=GREEN, lw=2.2)
ax.axhline(y.mean(), ls="--", color=_style.CB["grey"], lw=1.2,
           label=f"prevalence {y.mean():.4f}")
ax.scatter([sens], [prec], color=RED, zorder=5, s=60, label="80% TPR op. point")
ax.set_xlabel("Recall (sensitivity)")
ax.set_ylabel("Precision")
ax.set_title(f"Precision-recall (stack OOF), AP = {ap:.3f}")
ax.legend()
_style.save_vector(fig, f"{OUT}/fig8_pr_curve", dpi=DPI)
plt.close(fig)

# Metrics table
rows = [
    ("ROC-AUC", f"{auc:.4f}"),
    ("pAUC @ 80% TPR  (range [0, 0.20])", f"{pauc:.5f}"),
    ("Average precision (PR-AUC)", f"{ap:.4f}"),
    ("Specificity @ 80% TPR", f"{spec80:.4f}"),
    ("Specificity @ 90% TPR", f"{spec90:.4f}"),
    ("Specificity @ 95% TPR", f"{spec95:.4f}"),
    ("Precision @ 80% TPR", f"{prec:.4f}"),
    ("Confusion @ 80% TPR (TP/FP/FN/TN)", f"{TP}/{FP}/{FN}/{TN}"),
]
fig, ax = plt.subplots(figsize=(8.4, 4.4))
ax.axis("off")
tb = ax.table(cellText=rows, colLabels=["Metric (stack, OOF)", "Value"],
              cellLoc="left", loc="center")
tb.auto_set_font_size(False)
tb.set_fontsize(12)
tb.scale(1, 1.9)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor("#222222")
        cell.set_text_props(color="white", fontweight="bold")
    elif r % 2 == 1:
        cell.set_facecolor("#f3f3f3")
fig.tight_layout()
fig.savefig(f"{OUT}/fig9_metrics_table.svg")
fig.savefig(f"{OUT}/fig9_metrics_table.png", dpi=DPI, bbox_inches="tight")
plt.close(fig)

# Ranked prediction examples (quality-filtered crops)
try:
    d2 = d.reset_index(drop=True)
    order = np.argsort(-p)
    score = dict(zip(d2.isic_id, p))
    h5 = h5py.File("data/train-image.hdf5", "r")

    def crop(iid):
        raw = h5[iid][()]
        b = raw.tobytes() if isinstance(raw, np.ndarray) else bytes(raw)
        return np.array(Image.open(io.BytesIO(b)).convert("RGB"))

    tp_cand = [d2.isic_id[i] for i in order if y[i] == 1]
    fp_cand = [d2.isic_id[i] for i in order if y[i] == 0]
    tp_ids = _style.select_clean(crop, tp_cand, 6)
    fp_ids = _style.select_clean(crop, fp_cand, 6)

    fig, axes = plt.subplots(2, 6, figsize=(14, 5.2))
    for row, ids, col in [(0, tp_ids, GREEN), (1, fp_ids, RED)]:
        for ax, iid in zip(axes[row], ids):
            ax.imshow(crop(iid))
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(col); s.set_linewidth(2.5)
            ax.set_title(f"p={score[iid]:.2f}", fontsize=12, color=col)
    axes[0, 0].set_ylabel("true malignant", fontsize=13, fontweight="bold", color=GREEN)
    axes[1, 0].set_ylabel("false positive", fontsize=13, fontweight="bold", color=RED)
    fig.suptitle("Highest-scored lesions: caught malignancies (top) vs hardest benigns (bottom)",
                 fontsize=15, y=0.99)
    h5.close()
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(f"{OUT}/fig10_top_predictions.png", dpi=DPI)
    plt.close(fig)
    print("wrote fig10_top_predictions.png")
except Exception as e:
    print("prediction-example grid skipped:", repr(e))

print("DONE perf-extra figures ->", OUT)
