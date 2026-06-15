# Changelog

All notable changes to this project are documented here. Format: Keep a Changelog; SemVer.

## [Unreleased]
### Added
- Validation spine (`src/cv.py`): official pAUC@80%TPR + patient-grouped StratifiedGroupKFold
  with a hard no-leak guarantee. Proven numerically identical to the vendored official metric.
- Tabular expert (LightGBM) + intrinsic, leak-free feature engineering.
- Small pretrained image experts across an efficiency frontier (MobileNetV4 → MobileNetV5-300M).
- Efficiency axis (params / FLOPs / CPU latency) and Pareto plotting.
- Multi-agent workflow (cv-guardian, tabular-fe, vision-expert, efficiency-auditor, repro).
- CI (ruff + pytest), pre-commit, issue/PR templates, citation metadata.
- Conda env spec (`environment.yml`, env `isic2024`, python 3.12) with a CPU-only torch fallback.
- `make data` target: Kaggle competition download + unzip into `data/`.
- Quarto deliverables site under `site/` (index / methods / results / team + reveal.js slides),
  rendered to `docs/site/`; `make site`, `make slides`, `make env-export` targets.
- `REPRODUCE.md`: end-to-end re-run instructions (env → data → folds → models → frontier → site).

### Changed
- Migrated the dev/setup toolchain from uv to conda (`setup` target, README, CONTRIBUTING).
- Credited all three Pakistan.AI members (Raja M11217073, Sultan M11217078, Hassan M11217081)
  in `CITATION.cff`, README, and the site.
