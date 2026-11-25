import sys,json
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1])); from methods.adaptive_prompt import AdaptiveConnectivityPrompt
from experiments.explore_quick import loadrep
from experiments.run_all import metrics
rows=[]
for name in ['ADNI','ADNI_fMRI','FTD_fMRI','OCD_fMRI','PPMI']:
 reps,y,cl=loadrep(name,'data'); reps={k:v for k,v in reps.items() if k in (['raw','summary_conn'] if 'fMRI' in name else ['raw'])}; sp=json.load(open(Path('splits')/(name+'.json'))); vals=[]; selected=[]
 for f in sp['folds']:
  tr=np.array(f['train']); te=np.array(f['test']); m=AdaptiveConnectivityPrompt().fit_predict(reps,y,tr,te,len(cl)); vals.append(metrics(y[te],m,len(cl))['auc'])
  # rerun selection metadata is not retained by current helper output, record known reps
  selected.append('adaptive')
 row={'dataset':name,'auc_mean':float(np.mean(vals)),'auc_std':float(np.std(vals,ddof=1))}; print(row); rows.append(row)
pd.DataFrame(rows).to_csv('results/adaptive_prompt.csv',index=False)
