# Locked decisions (and the reasoning)

1. **Scope = single-dataset, no external data.** Train on ISIC-2024 SLICE-3D only
   (393 malignant / 400,666 benign). SLICE-3D from the ISIC Archive == the Kaggle train
   set; using it is not "external". The hidden test set is not public.
2. **No generative augmentation.** Medical images: training/validating on diffusion-made
   pathology is a label-validity hole. Classical augmentation only.
3. **Goal = quality-vs-cost Pareto frontier, not the absolute private-LB number.** The
   ~0.173 winners used external dermoscopy + ~30k synthetic positives, both banned here;
   matching that number is out of reach by construction. Claim: SOTA among single-dataset,
   no-external-data, no-synthetic solutions, reported on a frontier.
4. **GBDT-first; image expert earns its place.** Signal lives in the tabular metadata;
   LightGBM is the efficient frontier for it. Image backbone -> OOF prob + embedding stacked
   into the GBDT, kept only if CV lifts pAUC.
5. **Combiner stays trivial.** Rank/logit average or a light meta-LightGBM. A learned
   per-lesion gate ("MoE-style") runs only as an ablation; at 393 positives it likely loses
   to the trivial combiner — that's a reportable result.
6. **Pretrained weights allowed; external training data not.** ImageNet has no skin-cancer
   labels, so a pretrained encoder doesn't contaminate the no-external-data claim.
7. **Metric pinned at pAUC@80%TPR**, computed only by src/cv.py. Range [0,0.20].
8. **Validation is sacred.** Patient-grouped StratifiedGroupKFold, frozen once, shared by
   every model. cv-guardian has veto power. This is the anti-shakeup discipline.
9. **Venue = medical-imaging** (MICCAI ISIC workshop / Medical Image Analysis / npj Digital
   Medicine), not ML-flagship. The efficiency-frontier framing is what gives it teeth.
