# Companion papers & citations

Verify exact bibinfo before publication; entries below are accurate to the sources found.

## Dataset (cite both, per the license)
- **SLICE-3D dataset** — Kurtansky, N. et al. *The SLICE-3D dataset: 400,000 skin lesion
  image crops extracted from 3D TBP for skin cancer detection.* Scientific Data (2024).
  doi:10.1038/s41597-024-03743-w. Also cite: International Skin Imaging Collaboration,
  *SLICE-3D 2024 Challenge Dataset*, doi:10.34970/2024-slice-3d (CC BY-NC 4.0).
- **Challenge recap** — Kurtansky, N. et al. *Automated triage of cancer-suspicious skin
  lesions with 3D total-body photography.* npj Digital Medicine (2025). PMC12639164.
  (Establishes the intra-patient-context result you are deliberately NOT using external
  data to beat — useful framing for the no-external-data scope.)

## Metric
- Official scoring: ISIC pAUC-aboveTPR (Kaggle code/metric/isic-pauc-abovetpr) and
  github.com/ISIC-Research/Challenge-2024-Metrics. pAUC above 80% TPR, scores in [0,0.20].
  Note: the repo README states 88% ([0,0.12]); the live leaderboard used 80%. Pin 80%.

## Efficient backbones (the frontier)
- **MobileNetV4** — Qin, D. et al. *MobileNetV4: Universal Models for the Mobile Ecosystem.*
  arXiv:2404.10518 (2024).
- **MobileNetV5 / Gemma 3n encoder** — Google (2025); weights timm/mobilenetv5_300m.gemma3n;
  added in timm 1.0.16 (Wightman). Heavy anchor of the frontier (~300M params, 256-768px).
- **timm** — Wightman, R. *PyTorch Image Models.* github.com/huggingface/pytorch-image-models.
  (Source of FasterNet, SHViT, StarNet, GhostNetV3, EfficientViT, FastViT used in the sweep,
  plus the RTX-class inference-timing benchmark CSVs for the latency axis.)

## Imbalance & (partial) AUC optimization
- **Focal loss** — Lin, T.-Y. et al. *Focal Loss for Dense Object Detection.* ICCV (2017).
- **Deep AUC Maximization** — Yuan, Z. et al. *Large-scale Robust Deep AUC Maximization:
  A New Surrogate Loss and Empirical Studies on Medical Image Classification.* ICCV (2021).
- **Stochastic AUC max with DNNs** — Liu, M. et al. arXiv:1908.10831.
- **AUC consistency** — Gao, W. & Zhou, Z.-H. *On the Consistency of AUC Pairwise
  Optimization.* arXiv:1208.0645.
- **AUC maximization survey** — arXiv:2203.15046 (partial-AUC / FPR-restricted formulations).

## GBDT baselines
- **LightGBM** — Ke, G. et al. NeurIPS (2017).
- **CatBoost** — Prokhorenkova, L. et al. NeurIPS (2018).
- **XGBoost** — Chen, T. & Guestrin, C. KDD (2016).

## Related ISIC-2024 solutions (positioning)
- Hasan, M. Z. & Rifat, F. Y. *Hybrid Ensemble of Segmentation-Assisted Classification and
  GBDT ...* arXiv:2506.03420 (2025). (Image+GBDT+synthetic; explicitly the synthetic-augment
  approach you reject — good contrast for the no-synthetic-data argument.)
- *Skin Lesion Phenotyping via Nested Multi-modal Contrastive Learning.* arXiv:2505.23709
  (uses SLICE-3D + PAD-UFES; multimodal contrastive — contrast for single-dataset scope).
