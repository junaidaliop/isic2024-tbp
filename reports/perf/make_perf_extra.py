"""Extra performance figures for the deck: confusion matrix, PR curve, a metrics
table, and ranked prediction examples. Computed from the canonical stack OOF."""
import matplotlib
matplotlib.use("Agg")
import io

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from src import cv

OUT = "reports/perf"
GREEN = "#046A38"

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
print(f"specificity @90%TPR={spec90:.4f}  @95%TPR={spec95:.4f}")

# --- confusion matrix at the 80% TPR screening operating point ---
fig, ax = plt.subplots(figsize=(4.6, 4.2))
cm = np.array([[TN, FP], [FN, TP]])
ax.imshow(cm, cmap="Greens")
for (i, j), v in np.ndenumerate(cm):
    pct = v / cm.sum() * 100
    ax.text(j, i, f"{v:,}\n{pct:.1f}%", ha="center", va="center",
            fontsize=12, color="white" if (i == j) else "#222", fontweight="bold")
ax.set_xticks([0, 1], ["pred benign", "pred malignant"])
ax.set_yticks([0, 1], ["benign", "malignant"])
ax.set_title(f"Confusion matrix @ 80% TPR\n(stack OOF; sens {sens:.0%}, spec {spec80:.1%})",
             fontsize=11)
fig.tight_layout()
fig.savefig(f"{OUT}/fig7_confusion_matrix.png", dpi=150)
plt.close(fig)

# --- precision-recall curve ---
pr, rc, _ = precision_recall_curve(y, p)
fig, ax = plt.subplots(figsize=(5.2, 4.2))
ax.plot(rc, pr, color=GREEN, lw=2)
ax.axhline(y.mean(), ls="--", color="#999", lw=1, label=f"prevalence {y.mean():.4f}")
ax.scatter([sens], [prec], color="#E8392B", zorder=5, label="80% TPR op. point")
ax.set_xlabel("Recall (sensitivity)")
ax.set_ylabel("Precision")
ax.set_title(f"Precision-Recall (stack OOF) · AP = {ap:.3f}", fontsize=11)
ax.legend(fontsize=8)
fig.tight_layout()
fig.savefig(f"{OUT}/fig8_pr_curve.png", dpi=150)
plt.close(fig)

# --- metrics table ---
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
fig, ax = plt.subplots(figsize=(6.4, 3.2))
ax.axis("off")
tb = ax.table(cellText=rows, colLabels=["Metric (stack, OOF)", "Value"],
              cellLoc="left", loc="center")
tb.auto_set_font_size(False)
tb.set_fontsize(10)
tb.scale(1, 1.5)
for (r, c), cell in tb.get_celld().items():
    if r == 0:
        cell.set_facecolor(GREEN)
        cell.set_text_props(color="white", fontweight="bold")
fig.tight_layout()
fig.savefig(f"{OUT}/fig9_metrics_table.png", dpi=150)
plt.close(fig)

# --- ranked prediction examples from the crops ---
try:
    d2 = d.reset_index(drop=True)
    order = np.argsort(-p)
    tp_ids = [d2.isic_id[i] for i in order if y[i] == 1][:6]          # confident true malignant
    fp_ids = [d2.isic_id[i] for i in order if y[i] == 0][:6]          # top false positives
    score = dict(zip(d2.isic_id, p))
    h5 = h5py.File("data/train-image.hdf5", "r")

    def crop(iid):
        raw = h5[iid][()]
        b = raw.tobytes() if isinstance(raw, np.ndarray) else bytes(raw)
        return np.array(Image.open(io.BytesIO(b)).convert("RGB"))

    fig, axes = plt.subplots(2, 6, figsize=(12, 4.4))
    for ax, iid in zip(axes[0], tp_ids):
        ax.imshow(crop(iid)); ax.axis("off")
        ax.set_title(f"p={score[iid]:.2f}", fontsize=9, color=GREEN)
    for ax, iid in zip(axes[1], fp_ids):
        ax.imshow(crop(iid)); ax.axis("off")
        ax.set_title(f"p={score[iid]:.2f}", fontsize=9, color="#E8392B")
    axes[0, 0].set_ylabel("true malignant", fontsize=10)
    axes[1, 0].set_ylabel("false positive", fontsize=10)
    fig.suptitle("Highest-scored lesions: caught malignancies (top) vs hardest benigns (bottom)",
                 fontsize=11)
    h5.close()
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig10_top_predictions.png", dpi=150)
    plt.close(fig)
    print("wrote fig10_top_predictions.png")
except Exception as e:
    print("prediction-example grid skipped:", repr(e))

print("DONE perf-extra figures ->", OUT)
