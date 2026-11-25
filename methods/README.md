# Methods

`soft_prompt.py` implements Class-Conditional Soft Prompt (CCSP). A fold-local standardized PCA projection is frozen. For each class, K prompt vectors are optimized by balanced cross-entropy on class scores computed from cosine similarity, with an anchor penalty toward the training-fold prototype. No outer test examples enter the fit.
