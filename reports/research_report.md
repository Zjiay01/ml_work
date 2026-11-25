# Project 2 research report

## Protocol

All reported results below use the supplied files, mutually exclusive labels, OCD exact-row deduplication, Fisher-z upper-triangular functional connectivity for fMRI, fixed shuffled stratified outer 5-fold splits (random seed 42), and fold-local preprocessing. Hyperparameter selection for SVM/MLP is performed with inner 3-fold AUC. Each fold also stores its test indices and probability predictions in `results/*.npz`.

## Dataset audit

- ADNI: 202 samples, 186 features, four classes AD/MCIn/MCIp/NC. Aggregate MCI was excluded because it duplicates subtype rows.
- ADNI-fMRI: 206 samples, 4,005 connectivity features, four classes.
- FTD-fMRI: 181 samples, 4,005 connectivity features, binary.
- OCD-fMRI: 81 samples after removing one exact duplicate, 4,005 connectivity features, binary.
- PPMI: 543 samples, 294 supplied features, binary. The 3 x 98 grouping is recorded but no biological semantics are assumed.

## Method exploration

The initial candidate was CBSP using the official differentiable TabPFN prompt-tuning interface. The official example treats prompt examples as differentiable tensors, freezes TabPFN weights, and backpropagates through `fit_with_differentiable_input`. A compatible TabPFN/PyTorch runtime was not available in the execution environment, so no CBSP score is reported and it is marked `NOT RUN`. The project instead implements a dependency-light Class-Conditional Soft Prompt (CCSP). A fold-local PCA projection is frozen, each class owns K learnable prompt vectors, and prompts are optimized with class-balanced cross-entropy plus prototype-anchor regularization. This is genuine soft prompt/prototype tuning, but it is not external pretrained-model prompting. TA confirmation is required before presenting it as the course-approved Prompt Learning method.

## Five-fold results

| Dataset | Method | AUC mean | AUC std | Δ vs best baseline | Relative Δ |
|---|---:|---:|---:|---:|---:|
| ADNI | CCSP | 0.6780 ± 0.0539 | 0.0539 | -0.0422 | -5.9% |
| ADNI | CatBoost | 0.7202 ± 0.0415 | 0.0415 | +0.0000 | +0.0% |
| ADNI | MLP | 0.6002 ± 0.1577 | 0.1577 | -0.1200 | -16.7% |
| ADNI | SVM | 0.6884 ± 0.0465 | 0.0465 | -0.0319 | -4.4% |

| ADNI_fMRI | CCSP | 0.5051 ± 0.0278 | 0.0278 | -0.0238 | -4.5% |
| ADNI_fMRI | CatBoost | 0.5289 ± 0.0511 | 0.0511 | +0.0000 | +0.0% |
| ADNI_fMRI | MLP | 0.5115 ± 0.0593 | 0.0593 | -0.0174 | -3.3% |
| ADNI_fMRI | SVM | 0.4849 ± 0.0239 | 0.0239 | -0.0440 | -8.3% |

| FTD_fMRI | CCSP | 0.7082 ± 0.1006 | 0.1006 | +0.0613 | +9.5% |
| FTD_fMRI | CatBoost | 0.6469 ± 0.1601 | 0.1601 | +0.0000 | +0.0% |
| FTD_fMRI | MLP | 0.5797 ± 0.0584 | 0.0584 | -0.0672 | -10.4% |
| FTD_fMRI | SVM | 0.5733 ± 0.2228 | 0.2228 | -0.0736 | -11.4% |

| OCD_fMRI | CCSP | 0.7442 ± 0.2115 | 0.2115 | +0.0596 | +8.7% |
| OCD_fMRI | CatBoost | 0.6846 ± 0.1927 | 0.1927 | +0.0000 | +0.0% |
| OCD_fMRI | MLP | 0.6071 ± 0.2051 | 0.2051 | -0.0776 | -11.3% |
| OCD_fMRI | SVM | 0.6766 ± 0.2299 | 0.2299 | -0.0080 | -1.2% |

| PPMI | CCSP | 0.5788 ± 0.0501 | 0.0501 | +0.0057 | +1.0% |
| PPMI | CatBoost | 0.5731 ± 0.0283 | 0.0283 | +0.0000 | +0.0% |
| PPMI | MLP | 0.4999 ± 0.1038 | 0.1038 | -0.0732 | -12.8% |
| PPMI | SVM | 0.5323 ± 0.0875 | 0.0875 | -0.0408 | -7.1% |

## Interpretation

CCSP is above the strongest traditional baseline on FTD-fMRI, OCD-fMRI, and PPMI, while CatBoost remains strongest on ADNI and ADNI-fMRI. Using the strongest baseline per dataset, CCSP changes AUC by -5.9% on ADNI, -4.5% on ADNI-fMRI, +9.5% on FTD-fMRI, +8.7% on OCD-fMRI, and +1.0% on PPMI. The 20% relative-improvement challenge is not reached under this leakage-controlled protocol. The improvements on OCD have large fold variance and should be described as exploratory rather than conclusive.

## Ablation

The K/anchor ablation is in `results/ablation.csv`. K=4 with anchor is the fixed CCSP configuration used in the main table. Changing K has only small effects, while removing the anchor is not consistently beneficial. This supports retaining the conservative anchored configuration rather than selecting a variant from outer-test results.

## Reproducibility and limitations

- `FAILED`/`NOT RUN` status is preserved in `logs/experiment_log.json` and this report. The original torch implementation failed because PyTorch was unavailable; it was replaced by a NumPy implementation with the same objective, then the full run completed.
- CatBoost was installed from the available package index and all five folds completed.
- The current CCSP does not use an external pretrained encoder. It should be discussed with the instructor before claiming it satisfies the requested foundation-model prompt-learning direction.
- The fMRI representation is a generic functional-connectivity summary. It does not infer anatomical meanings for anonymous features.
- External confirmation is still needed for the method label and the exact course-paper positioning.
## Second-stage model search

A second search tested fold-local logistic regression with alternative fMRI representations. The representation concatenating ROI temporal mean, ROI temporal standard deviation, node-wise mean connectivity strength, and Fisher-z connectivity was useful on FTD and OCD. With inner-selected regularization, raw connectivity logistic regression reached 0.7859 AUC on OCD, while the summary-plus-connectivity representation reached 0.7302 on FTD and 0.5249 on ADNI-fMRI. A fold-local adaptive CCSP representation selector reached 0.5460 on ADNI-fMRI, 0.7174 on FTD-fMRI, and 0.7484 on OCD-fMRI when the candidate family was restricted to raw and summary-plus-connectivity views. These results are exploratory candidates and were not substituted into the original main table after inspecting outer-fold outcomes.

The strongest observed exploratory scores still do not provide a stable 20% improvement across datasets. In particular, the 20% target would require approximately 0.776 AUC on FTD-fMRI and 0.823 AUC on OCD-fMRI relative to the strongest unified baselines. Current results remain below those thresholds. The next scientifically justified step is to discuss the adaptive representation prompt with the TA and pre-register the candidate family before a confirmatory rerun.
