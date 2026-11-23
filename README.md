# Project 2: Prompt-aware small-sample neuroimaging classification

This repository contains the reproducible Project 1/2 pipeline for the five supplied datasets. The evaluation protocol is a fixed stratified outer 5-fold split with fold-local preprocessing and inner 3-fold model selection. Outer test folds are never used for fitting or tuning.

## Status

- Data inspection: `RUN`
- Unified splits: `RUN`
- SVM / MLP / CatBoost baselines: `RUN` or `NOT RUN` in `results/summary.csv`
- CBSP with TabPFN: `NOT RUN` unless the optional TabPFN dependency is available. The implementation records this explicitly.
- Prompt method: `ClassConditionalSoftPrompt` in `methods/soft_prompt.py`

## Reproduce

```bash
python data_inspection/inspect_mat.py --data-dir data --out-dir results/inspection
python preprocessing/make_splits.py --data-dir data --out-dir splits
python experiments/run_all.py --data-dir data --split-dir splits --out-dir results --log-dir logs
python experiments/summarize.py --results-dir results
```

The scripts write out-of-fold probabilities, fold metrics, aggregate metrics, method exploration logs, and a report draft. Dataset files are included locally for reproducibility of this course project.
