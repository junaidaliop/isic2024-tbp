<div align="center">

# ISIC-2024 SLICE-3D — Skin-Cancer Detection on a Quality–Cost Frontier

**A single-dataset, no-external-data, no-synthetic study of melanoma triage from 3D total-body photography.**

[![ci](https://github.com/junaidaliop/isic2024-tbp/actions/workflows/ci.yml/badge.svg)](https://github.com/junaidaliop/isic2024-tbp/actions/workflows/ci.yml)
[![python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](pyproject.toml)
[![code: MIT](https://img.shields.io/badge/code-MIT-046A38.svg)](LICENSE)
[![data: CC BY-NC 4.0](https://img.shields.io/badge/data-CC%20BY--NC%204.0-lightgrey.svg)](docs/CITATIONS.md)
<br>
[![OOF pAUC@80%TPR](https://img.shields.io/badge/OOF%20pAUC%4080%25TPR-0.17376-046A38.svg)](https://junaidaliop.github.io/isic2024-tbp/results.html)
[![constraint](https://img.shields.io/badge/no%20external%20%C2%B7%20no%20synthetic-single--dataset-034425.svg)](#the-question)
[![validation](https://img.shields.io/badge/CV-patient--grouped%20%C2%B7%20leak--audited-046A38.svg)](#why-the-numbers-are-trustworthy)

**[Companion site](https://junaidaliop.github.io/isic2024-tbp/)** · **[Interactive slides](https://junaidaliop.github.io/isic2024-tbp/slides.html)** · **[Methods](https://junaidaliop.github.io/isic2024-tbp/methods.html)** · **[Results](https://junaidaliop.github.io/isic2024-tbp/results.html)** · **[Ablations](https://junaidaliop.github.io/isic2024-tbp/ablations.html)**

</div>

---

## TL;DR

The ISIC-2024 winners reached pAUC@80%TPR ≈ 0.173 — but only by importing external dermoscopy
archives **and** ~30,000 diffusion-synthesised malignant lesions. We ban both and ask a sharper
question: **how much pAUC can you buy per unit of inference cost using SLICE-3D alone?** The answer
is a leak-audited **quality–cost Pareto frontier** whose headline point, a parameter-free
rank-average of a bagged-GBDT tabular expert and a small ConvNeXt-V2 image expert, reaches
**out-of-fold pAUC@80%TPR = 0.17376** at 15.8 M params / 2.46 GFLOPs / 61 ms single-thread CPU.

> **Claim.** State of the art *among single-dataset, no-external, no-synthetic* solutions, reported
> on a transparent quality–efficiency frontier rather than a single number. The unconstrained
> ~0.173 is out of reach by construction; quantifying that gap honestly is the contribution.

## The question

ISIC-2024 SLICE-3D is melanoma **triage**: rank 401,059 lesion crops from 3D total-body photography
(393 malignant, 0.098 % prevalence, ~1,042 patients) so the highest-risk are reviewed first, scored
by the official **partial AUC above 80 % TPR** (range `[0, 0.20]`; random ≈ 0.02, perfect = 0.20).
We impose three rules and report what they cost:

- **One dataset.** SLICE-3D only — no ISIC-2019/2020, no PAD-UFES, no external dermoscopy.
- **No synthetic pathology.** No diffusion/GAN-fabricated lesions; classical augmentation only.
- **Efficiency is a first-class axis.** Every model logs params, FLOPs, and CPU latency, so each is
  a measured point on a Pareto frontier — not just an accuracy number.

ImageNet-pretrained encoders are allowed (they carry no skin-cancer labels, so the no-external-*data*
claim survives); external *training data* is not.

## Headline results

All numbers are **out-of-fold** on the frozen patient-grouped folds, scored only by `src/cv.py`, and
independently re-derived from disk.

| Model | pAUC@80%TPR | Params (M) | GFLOPs | CPU ms* | Role |
|---|---:|---:|---:|---:|---|
| Tabular — LightGBM + CatBoost (bagged) | **0.16890** | 0.86 | ~0 | 0.02 | near-free anchor |
| Image — ConvNeXt-V2-nano @224 | 0.15821 | 14.98 | 2.46 | 60.9 | primary backbone |
| **Stack — rank-avg(tabular, image)** | **0.17376** | 15.84 | 2.46 | 60.9 | **best; Pareto-optimal** |
| Cheap stack — gbdt + 3 images + ugly-duckling | 0.17117 | 23.84 | 1.22 | 32.9 | cheaper frontier point |
| EfficientViT-b0 @128 | 0.13706 | 2.13 | 0.034 | 3.5 | most accuracy per ms |

\* median single-image latency, one thread.

![Quality–cost Pareto frontier: OOF pAUC@80%TPR vs parameters, GFLOPs, and single-thread CPU latency.](docs/figures/frontier.png)

## What this project shows

- **A no-external/no-synthetic frontier, not a point.** Twelve backbones plus the tabular and
  stacked models are each measured on three cost axes, drawing the achievable quality-vs-cost curve.
- **The signal is patient-relative.** Engineered *ugly-duckling* deviations — how a lesion departs
  from the **same patient's** other moles — account for **~65 %** of the GBDT's gain. Hue alone
  reaches univariate AUC 0.81.
- **Trivial fusion wins at 393 positives.** A zero-parameter rank-average beats a meta-LightGBM
  stacker, a learned per-lesion gate, and embedding injection — a clean, reportable result about the
  small-positive regime, not a footnote.
- **Leak-audited from the spine outward.** One frozen split, one metric function proven identical to
  the official scorer, a `cv-guardian` with veto power, and a 9/9 test suite.

## Honest negative results

Nearly every increase in model complexity *reduced* the score. We log them rather than hide them.

| Idea tried | Its pAUC | Beaten by | Finding |
|---|---:|---|---|
| Learned per-lesion gate (MoE) | 0.15007 | 0.16890 (tabular) | loses to the tabular expert alone |
| Meta-LightGBM stacker | 0.17108 | 0.17376 (rank-avg) | loses to the trivial rank-average |
| PCA image embeddings into the GBDT | < 0.17376 | 0.17376 | extra dimensions overfit and dilute |
| Heavy transformers (SwinV2 / EVA-02) | 0.104 / 0.100 | 0.15821 (nano@224) | collapse to near-random at 393 positives |
| Stronger weight EMA (0.999) | ~0.146 | 0.15821 (EMA 0.995) | over-smooths the short undersampled epochs |

## Method

Two experts and a deliberately trivial combiner, all reading one frozen validation split.

- **Tabular expert** (`src/features.py`, `src/gbdt.py`). From ISIC-2024 metadata only: lesion
  geometry and size ratios, L\*/A\*/B\* colour and lesion-vs-skin contrast, border/shape composites,
  3D body position, and patient-relative deviations (`pdev_`/`prank_`/`pxc_`) with fold-local target
  encoding. Bagged LightGBM (5 seeds) + CatBoost, manual undersampling, pAUC early-stopping → **0.16890**.
- **Image expert** (`src/vision/`). Small ImageNet-pretrained backbones (ConvNeXt-V2-nano the
  optimum) at 128–224 px. AdamW (1e-4, wd 1e-3), cosine schedule, 30 epochs, batch 128; per-epoch
  negative undersampling (~1:1); BCE + label smoothing 0.05; classical `transV2` augmentation (flips,
  brightness/contrast, blur/noise, optical/grid/elastic distortion, CLAHE, hue–saturation,
  shift–scale–rotate, coarse-dropout; **no generative augmentation**); weight EMA 0.995 and mixup
  α = 0.2. Emits an OOF probability + embedding → **0.15821**.
- **Combiner** (`src/stack.py`). A parameter-free rank-average of the two OOF probabilities → **0.17376**.

## Why the numbers are trustworthy

Leaky cross-validation moved teams ~200 places on the private split in this competition, so
validation is the foundation here, not an afterthought.

- `src/cv.py` is the single source of truth for the metric and the **patient-grouped,
  target-stratified** 5-fold split (SEED 42); no patient straddles folds.
- `tests/test_cv.py` proves `src/cv.py` is numerically identical to the vendored official scorer
  (`src/metric_official.py`, © 2024 N. R. Kurtansky, MSKCC).
- Every model reads the same frozen `data/folds.parquet`; every patient-relative statistic and target
  encoding is computed fold-locally. Tabular and folds reproduce bit-exact.

## Reproduce

```bash
conda create -y -n isic2024 python=3.12 && conda activate isic2024
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8
pip install -e ".[dev]"                                                            # (CPU: use .../whl/cpu)

make data       # download SLICE-3D into data/ (accept the Kaggle rules first)
make folds      # freeze patient-grouped folds + metric sanity — run this first
make test       # spine tests: metric anchors, no-leak, official-equivalence
make gbdt       # bagged tabular model            -> OOF + pAUC  (0.16890)
make vision CFG=configs/vision/convnextv2_nano_r224.yaml   # image expert -> OOF + embeddings (0.15821)
python experiments/run_stack.py    # rank-average -> stack OOF (0.17376) + frontier row
make frontier   # Pareto figures (pAUC vs params / GFLOPs / CPU ms)
make site       # render the companion website -> docs/
```

Single seed (42) everywhere; runs are config-driven (`configs/`) and logged; the environment is
pinned (`requirements-lock.txt` / `environment.yml`). Full grader sequence in [`REPRODUCE.md`](REPRODUCE.md).

## Submission (closed competition)

ISIC-2024 was a Kaggle *code* competition: the private score comes only from a notebook run on the
hidden test set. `src/submit.py` builds a format-valid `submission.csv` from the saved tabular models;
the CV-estimated private score is ~0.155–0.165, competitive with the best no-external/no-synthetic
entries. See the [submission page](https://junaidaliop.github.io/isic2024-tbp/submission.html).

## Repository

```
src/cv.py            validation spine: official pAUC + frozen patient-grouped folds
src/metric_official  vendored official ISIC-2024 scorer (verbatim, attributed)
src/features.py      intrinsic, leak-free tabular feature engineering
src/gbdt.py          bagged LightGBM/CatBoost expert + OOF
src/vision/          small pretrained image experts + OOF/embeddings
src/efficiency.py    params / FLOPs / CPU latency
src/stack.py         combiner + frontier logging
src/submit.py        submission builder
experiments/         the stack ablation + headline OOF (run_stack.py)
reports/             frontier + EDA + performance figure generators
site/  ·  docs/      Quarto companion website (source · rendered for GitHub Pages)
```

## Team — Pakistan.AI

Neural Networks course project, National Yunlin University of Science and Technology.
Instructor: **Prof. Hsuan-Ting Chang**.

| Member | Student ID |
|---|---|
| Raja, Muhammad Junaid Ali Asif ([ORCID 0009-0008-9249-9983](https://orcid.org/0009-0008-9249-9983)) | M11217073 |
| Sultan, Adil | M11217078 |
| Hassan, Shahzaib Ahmed | M11217081 |

## Citing & license

Cite via [`CITATION.cff`](CITATION.cff) (GitHub renders a "Cite this repository" button); please also
cite the SLICE-3D dataset and the official metric ([`docs/CITATIONS.md`](docs/CITATIONS.md)). Code is
MIT ([`LICENSE`](LICENSE)); the data is **not** included and is CC BY-NC 4.0 — obtain and attribute it
separately.
