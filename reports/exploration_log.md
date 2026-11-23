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
