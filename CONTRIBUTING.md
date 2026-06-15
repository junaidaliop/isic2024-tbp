# Contributing

This is a research repository; reproducibility and validation integrity come before features.

## Non-negotiables
- **Never touch the CV contract casually.** Splits come only from `src/cv.py`; the metric is
  computed only by `src/cv.py:pauc_above_tpr` (proven equal to the vendored official metric).
  Any PR touching splits, the metric, or feature leakage must pass the `cv-guardian` review.
- **No external data, no synthetic/generated data.** See `docs/DECISIONS.md`.
- **Every reported pAUC ships with its cost** (params / FLOPs / CPU latency) and a frontier row.
- **Negative results are logged, not deleted.**

## Workflow
1. Branch from `main`. Keep PRs focused.
2. `pre-commit run --all-files` (ruff) and `pytest -q` must pass.
3. Any new number must be reproducible from a committed config + fixed seed (`repro` agent).
4. Describe the pAUC delta AND the cost delta in the PR.

## Dev setup
```bash
conda create -y -n isic2024 python=3.12
conda activate isic2024
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -e ".[dev]"
pip install kaggle pre-commit
pre-commit install
# CPU-only: swap the index for https://download.pytorch.org/whl/cpu
# one-shot alternative:  conda env create -f environment.yml
```
`pre-commit run --all-files` (ruff lint + format) and `pytest -q` must both pass before a PR.
