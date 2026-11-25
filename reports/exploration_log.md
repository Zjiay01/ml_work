# Method exploration log

## Dataset diagnosis

- ADNI uses the four mutually exclusive classes AD, MCIn, MCIp and NC. The aggregate MCI rows are excluded because they duplicate subtype rows.
- fMRI arrays are converted to 4,005 Fisher-z upper-triangular functional-connectivity features. Correlations are computed per subject before any fold-local scaling.
- OCD exact duplicate rows are removed before split generation. The resulting sample count is 81.
- PPMI is retained as 294 supplied features and recorded as three contiguous 98-feature groups for future ablation. The current first method uses all features to avoid inventing biological feature semantics.

## Candidate status

| Candidate | Status | Reason |
|---|---|---|
| SVM | RUN | Strong small-sample baseline with fold-local scaling/PCA and inner selection |
| MLP | RUN | Course baseline with fold-local feature selection and inner selection |
| CatBoost | RUN/FAILED | Fixed, auditable course baseline; package availability recorded in logs |
| CBSP + TabPFN | NOT RUN | Optional TabPFN dependency and version-compatible differentiable prompt API were not assumed without verification. No fabricated result is reported. |
| Class-Conditional Soft Prompt (CCSP) | RUN | Fold-local PCA is frozen and class-specific prompt vectors are optimized by balanced CE plus prototype anchor regularization |

CCSP is a genuine soft-prompt/prototype-tuning method, but it does not claim external pretraining. Its paper positioning must be confirmed with the instructor/TA before submission.

## Second-stage search

- Logistic regression on fold-local ROI summary plus connectivity representations improved ADNI-fMRI to 0.5532 with fixed C and improved FTD-fMRI to 0.7398 in exploratory fixed-parameter runs.
- Inner-selected logistic regression reached 0.7859 on OCD-fMRI and 0.7302 on FTD-fMRI. These are candidate-search results and are not the primary confirmatory table.
- Adaptive CCSP selected among fold-local raw and summary-plus-connectivity views. It reached 0.5460 on ADNI-fMRI, 0.7174 on FTD-fMRI, 0.7484 on OCD-fMRI, and 0.5788 on PPMI. Representation selection was unstable on ADNI-fMRI when the candidate family was too broad.
- Focused RBF-SVM search was started but terminated after excessive CPU time and is marked NOT RUN in the search log. No result from that run is used.
