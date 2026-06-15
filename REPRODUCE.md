# Reproducing the results

End-to-end re-run for a grader, from a clean checkout. Single seed (`SEED=42`) everywhere
stochastic; every number is config-driven and logged. If a re-run does not match, the `repro`
agent treats that as a bug, not noise.

Constraints that make the numbers meaningful (see `docs/DECISIONS.md`):
**single dataset** (ISIC-2024 SLICE-3D only — no ISIC-2019/2020, no PAD-UFES, no external
dermoscopy), **no generative/synthetic augmentation**, **patient-grouped folds frozen once and
shared by every model**, and the **official pAUC@80%TPR** (range `[0, 0.20]`) computed only by
`src/cv.py:pauc_above_tpr`.

## 0. Environment
```bash
conda create -y -n isic2024 python=3.12
conda activate isic2024
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.8
pip install -e ".[dev]"
pip install kaggle pre-commit
```
CPU-only graders (no GPU): swap the torch index for `https://download.pytorch.org/whl/cpu`.
One-shot alternative: `conda env create -f environment.yml`.

The spine tests (metric anchors, no-leak, official-equivalence) need no GPU and no torch —
plain `pip install numpy pandas pyarrow scikit-learn scipy pyyaml pytest ruff` is enough for
`make test`.

## 1. Data
Accept the competition rules once on the Kaggle website (otherwise the API returns 403), put
your `kaggle.json` token in `~/.kaggle/`, then:
```bash
make data        # kaggle competitions download -c isic-2024-challenge -p data/ && unzip
```
Expected in `data/` (gitignored, CC BY-NC 4.0 — not committed):
`train-metadata.csv`, `train-image.hdf5`, `test-metadata.csv`, `test-image.hdf5`,
`sample_submission.csv`. See `data/README.md` and `docs/CITATIONS.md`.

## 2. Freeze the CV spine (always first)
```bash
make folds       # writes data/folds.parquet (patient-grouped, target-stratified)
make test        # metric anchors + no-leak + numerical match to the official scorer
```
Nothing else should run until this passes. No patient may straddle folds; every later model
reads this exact `data/folds.parquet`.

## 3. Models
```bash
make gbdt                                       # LightGBM tabular expert -> OOF + pAUC
make vision CFG=configs/vision/mnv4_small.yaml  # small image expert -> OOF prob + embedding
```
Sweep additional vision configs to draw the frontier rather than hit one point. Each backbone
is kept only if cross-validation lifts pAUC over the GBDT-only baseline; negative results are
logged, not deleted.

## 4. Frontier + deliverables
```bash
make frontier    # reports/frontier.csv -> reports/frontier.png (pAUC vs cost)
make site        # renders the Quarto site to docs/site/
make slides      # renders the reveal.js deck to PDF
```
`reports/*.csv` and `reports/*.png` are gitignored, so run `make frontier` **before**
`make site` (or commit the rendered `docs/site/`); `site/results.qmd` degrades gracefully to a
"results pending" note when the CSV/PNG are absent.

## Seeds & provenance
`SEED=42` is fixed in every stochastic step. Each run carries its config under `configs/` and
a row in `experiments/` / `reports/frontier.csv`. To re-export the exact environment after any
change: `make env-export`.
