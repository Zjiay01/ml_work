import sys,json
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1])); from methods.soft_prompt import ClassConditionalSoftPrompt
from experiments.explore_quick import loadrep
from experiments.run_all import metrics
rows=[]
for name in ['ADNI_fMRI','FTD_fMRI','OCD_fMRI']:
 reps,y,cl=loadrep(name,'data'); sp=json.load(open(Path('splits')/(name+'.json')))
 for rep,X in reps.items():
  vals=[]
  for f in sp['folds']:
   tr=np.array(f['train']); te=np.array(f['test']); m=ClassConditionalSoftPrompt(n_prompts=4,n_components=32,epochs=180,lr=.03,anchor=.05,seed=42).fit(X[tr],y[tr]); vals.append(metrics(y[te],m.predict_proba(X[te]),len(cl))['auc'])
  print(name,rep,np.mean(vals),np.std(vals,ddof=1)); rows.append({'dataset':name,'rep':rep,'auc':np.mean(vals),'std':np.std(vals,ddof=1)})
pd.DataFrame(rows).to_csv('results/ccsp_representations.csv',index=False)
