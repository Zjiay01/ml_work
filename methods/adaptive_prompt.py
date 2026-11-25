import numpy as np
from sklearn.model_selection import StratifiedKFold
from .soft_prompt import ClassConditionalSoftPrompt
class AdaptiveConnectivityPrompt:
 """Selects a fold-local representation using inner AUC, then tunes CCSP prompts."""
 def __init__(self,epochs=180,seed=42): self.epochs=epochs; self.seed=seed
 def fit_predict(self,reps,y,tr,te,n_classes):
  names=list(reps); inner=StratifiedKFold(3,shuffle=True,random_state=123); scores={n:[] for n in names}
  for itr,iv in inner.split(reps[names[0]][tr],y[tr]):
   a=tr[itr]; b=tr[iv]
   for n in names:
    m=ClassConditionalSoftPrompt(n_prompts=4,n_components=min(32,reps[n].shape[1]),epochs=self.epochs,seed=self.seed).fit(reps[n][a],y[a]); p=m.predict_proba(reps[n][b]); from sklearn.metrics import roc_auc_score; scores[n].append(roc_auc_score(y[b],p[:,1] if n_classes==2 else p,multi_class='ovr' if n_classes>2 else 'raise',average='macro'))
  self.selected_rep_=max(names,key=lambda n:np.mean(scores[n])); self.inner_scores_={n:float(np.mean(v)) for n,v in scores.items()}; m=ClassConditionalSoftPrompt(n_prompts=4,n_components=min(32,reps[self.selected_rep_].shape[1]),epochs=self.epochs,seed=self.seed).fit(reps[self.selected_rep_][tr],y[tr]); return m.predict_proba(reps[self.selected_rep_][te])
