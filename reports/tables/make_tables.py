"""Academic booktabs-style HTML tables for ISIC-2024 SLICE-3D.

Reads canonical files only (reports/perf/_summary.json, reports/frontier_cost.csv,
reports/frontier.csv) and the leaderboard numbers from site/credits.qmd.
Writes reports/tables/table{1,2,3}.html. Run: PYTHONPATH=. python reports/tables/make_tables.py
"""
from __future__ import annotations

import html
import json
import os

import pandas as pd

ROOT = "/home/efreet/Ali/isic2024-tbp"
OUT = os.path.join(ROOT, "reports", "tables")
os.makedirs(OUT, exist_ok=True)

with open(os.path.join(ROOT, "reports", "perf", "_summary.json")) as f:
    SUM = json.load(f)
PAUC = SUM["pauc"]


def esc(x):
    return html.escape(str(x))


def write_table(fname, caption, headers, rows, num_cols, highlight=None, note=None):
    """headers: list[str]; rows: list[list]; num_cols: set[int] right-aligned."""
    highlight = highlight or set()
    h = []
    h.append('<link rel="stylesheet" href="tables.css">')
    h.append('<table class="acad-table">')
    h.append(f"  <caption>{esc(caption)}</caption>")
    h.append("  <thead><tr>")
    for j, head in enumerate(headers):
        cls = ' class="num"' if j in num_cols else ""
        h.append(f"    <th{cls}>{esc(head)}</th>")
    h.append("  </tr></thead>")
    h.append("  <tbody>")
    for i, row in enumerate(rows):
        tr = ' class="highlight"' if i in highlight else ""
        h.append(f"    <tr{tr}>")
        for j, cell in enumerate(row):
            cls = ' class="num"' if j in num_cols else ""
            h.append(f"      <td{cls}>{esc(cell)}</td>")
        h.append("    </tr>")
    h.append("  </tbody>")
    h.append("</table>")
    if note:
        h.append(f'<p class="note">{esc(note)}</p>')
    out = os.path.join(OUT, fname)
    with open(out, "w") as f:
        f.write("\n".join(h) + "\n")
    print("wrote", out)
    return out


# === TABLE 1 — tabular-only progression ===
tab_final = f"{PAUC['tabular (gbdt)']:.5f}"
t1_rows = [
    ["Step 0 — broken: is_unbalance + AUC early-stop", "0.09941"],
    ["Step 1 — fixed early-stopping / objective", "0.11826"],
    ["Step 2 — wide patient-relative ugly-duckling features", "0.14420"],
    ["Step 3 — greysky-bagged + cross feats + CatBoost ensemble", tab_final],
]
write_table(
    "table1_tabular.html",
    "Table 1. Tabular LightGBM progression (out-of-fold, leak-verified).",
    ["Configuration", "OOF pAUC@80%TPR"],
    t1_rows, num_cols={1}, highlight={3},
)

# === TABLE 2 — efficiency frontier (all experiments) ===
cost = pd.read_csv(os.path.join(ROOT, "reports", "frontier_cost.csv"))
fr = pd.read_csv(os.path.join(ROOT, "reports", "frontier.csv"))
best_pauc = fr.groupby("model")["pauc"].max().to_dict()

# Map frontier_cost model names -> a display name + a pAUC source.
ROW_PAUC = {
    "gbdt": PAUC["tabular (gbdt)"],
    "convnextv2_nano": PAUC["img:convnextv2_nano"],
    "convnextv2_nano_r224": PAUC["img:convnextv2_nano_r224"],
    "convnextv2_tiny": PAUC["img:convnextv2_tiny"],
    "effvit_b0": PAUC["img:effvit_b0"],
    "vit_tiny": PAUC["img:vit_tiny"],
    "mnv4_small": PAUC["img:mnv4_small"],
    "swinv2_tiny": PAUC["img:swinv2_tiny"],
    "eva02_small": PAUC["img:eva02_small"],
    "stack_best": PAUC["stack (rank-avg gbdt+r224)"],
    "stack_gbdt+3img+udk": best_pauc.get("stack_gbdt+3img+udk"),
}
DISPLAY = {
    "gbdt": "Tabular GBDT (LightGBM + CatBoost)",
    "convnextv2_nano": "ConvNeXtV2-nano @128",
    "convnextv2_nano_r224": "ConvNeXtV2-nano @224",
    "convnextv2_tiny": "ConvNeXtV2-tiny @224",
    "effvit_b0": "EfficientViT-b0 @128",
    "vit_tiny": "ViT-tiny @128",
    "mnv4_small": "MobileNetV4-small @128",
    "swinv2_tiny": "SwinV2-tiny @256",
    "eva02_small": "EVA-02-small @336",
    "stack_best": "Stack (rank-avg: GBDT + nano@224)",
    "stack_gbdt+3img+udk": "Stack (GBDT + 3 images + ugly-duckling)",
}

cost = cost[cost["model"].isin(ROW_PAUC)].copy()
cost["pauc"] = cost["model"].map(ROW_PAUC).astype(float)


def is_pareto(d):
    c = d["cpu_ms"].to_numpy(float)
    q = d["pauc"].to_numpy(float)
    mask = []
    for i in range(len(d)):
        dom = any(
            c[j] <= c[i] and q[j] >= q[i] and (c[j] < c[i] or q[j] > q[i])
            for j in range(len(d)) if j != i)
        mask.append(not dom)
    return mask


cost = cost.sort_values("pauc", ascending=False).reset_index(drop=True)
cost["pareto"] = is_pareto(cost)

t2_rows, hl = [], set()
for i, r in cost.iterrows():
    name = r["model"]
    t2_rows.append([
        DISPLAY[name],
        f"{r['pauc']:.5f}",
        f"{r['params_m']:.2f}",
        f"{r['gflops']:.4f}",
        f"{r['cpu_ms']:.2f}",
        "Yes" if r["pareto"] else "—",
    ])
    if name == "stack_best":
        hl.add(i)
write_table(
    "table2_experiments.html",
    "Table 2. Quality–efficiency frontier across all experiments "
    "(out-of-fold pAUC; single-thread CPU cost).",
    ["Model", "OOF pAUC@80%TPR", "Params (M)", "GFLOPs",
     "CPU latency (ms)", "Pareto-optimal"],
    t2_rows, num_cols={1, 2, 3, 4}, highlight=hl,
    note="CPU latency is single-thread, one image; GBDT GFLOPs reported as 0.0 "
         "(tree ensemble, not FLOP-comparable). Pareto-optimal on the pAUC-vs-latency axis.",
)

# === TABLE 3 — honest leaderboard comparison (numbers from site/credits.qmd) ===
t3_rows = [
    ["1st — Ilya Novoselskiy (EVA-02 + EdgeNeXt + GBDT)", "Yes", "Yes (~30k)", "0.17264"],
    ["2nd — uchiyama33 (image + tabular ensemble)", "Yes", "Yes", "—"],
    ["3rd — kyohei-123 (image + tabular blend)", "Yes", "Yes", "—"],
    ["Ours (single-dataset, no external, no synthetic)", "No", "No", "CV 0.17376"],
]
write_table(
    "table3_leaderboard.html",
    "Table 3. Honest comparison to top Kaggle solutions "
    "(champion private-LB pAUC; ours is leak-audited cross-validation).",
    ["Solution", "External data?", "Synthetic data?", "pAUC"],
    t3_rows, num_cols={3}, highlight={3},
    note="Champions used external ISIC-archive dermoscopy and synthetic positives, "
         "both banned here; their own ablation reports the ~30k synthetic lesions "
         "added only +0.0007 pAUC. Ours is an out-of-fold CV number, not private LB.",
)

print("DONE tables ->", OUT)
