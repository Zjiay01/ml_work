# Automatic research iteration log

The goal is a 20% relative AUC increase over the strongest unified baseline on every dataset. Each candidate is evaluated with the same fixed outer five-fold indices and fold-local preprocessing. Candidate selection uses training folds only.

| Version | Method | Main change | Best observed AUC | Status |
|---|---|---|---|---|
| V0 | SVM/MLP/CatBoost | Unified baselines | ADNI 0.7202, ADNI-fMRI 0.5289, FTD 0.6469, OCD 0.6846, PPMI 0.5731 | Baseline |
| V1 | CCSP | Class-conditional soft prompts over fold-local PCA | ADNI 0.6780, ADNI-fMRI 0.5051, FTD 0.7082, OCD 0.7442, PPMI 0.5788 | Mixed |
| V2 | Connectivity summaries | ROI mean/std and node-strength summaries | ADNI-fMRI 0.5532, FTD 0.7398, OCD 0.7641 | Exploratory |
| V3 | Regularized logistic search | Inner-selected C and PCA on raw/summary views | OCD 0.7859, FTD 0.7302, ADNI-fMRI 0.5630 | Best exploratory |
| V4 | Adaptive CCSP | Inner fold representation selection followed by prompt tuning | ADNI-fMRI 0.5460, FTD 0.7174, OCD 0.7484 | Unstable |
| V5 | LightGBM | Small-sample boosted-tree alternative | ADNI 0.6612, ADNI-fMRI 0.4871, FTD 0.6225, OCD 0.7112, PPMI 0.5366 | Rejected |
| V6 | TabPFN/CBSP runtime | Install official differentiable prompt stack | NOT RUN | Blocked by environment disk/runtime limits |

## Current decision

The best simple candidate is inner-selected regularized logistic regression with a single fold-local feature representation. It is simpler and stronger than the previous multi-stage prompt pipeline on fMRI cohorts. It still falls short of 20% on every dataset, so it should remain an exploratory candidate until a confirmatory rerun is pre-registered.
| V7 | Elastic-Net logistic | Single PCA plus sparse/ dense mixed regularization | ADNI 0.7271, ADNI-fMRI 0.5009, FTD 0.7146, OCD 0.7324, PPMI 0.5641 | Rejected |

V7 improved ADNI slightly but degraded the fMRI and PPMI cohorts, so it is not retained.
