import argparse,json,sys
from pathlib import Path
import numpy as np,pandas as pd
sys.path.insert(0,str(Path(__file__).resolve().parents[1])); from preprocessing.datasets import DATASETS,load_dataset
from methods.soft_prompt import ClassConditionalSoftPrompt
from experiments.run_all import metrics
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='../data'); ap.add_argument('--split-dir',default='../splits'); ap.add_argument('--out',default='../results/ablation.csv'); a=ap.parse_args(); rows=[]
 variants=[('K1_anchor',1,.05),('K4_anchor',4,.05),('K4_no_anchor',4,0.),('K8_anchor',8,.05)]
 for name in DATASETS:
  X,y,classes,_=load_dataset(name,a.data_dir); spec=json.load(open(Path(a.split_dir)/(name+'.json')))
  for var,k,anchor in variants:
   vals=[]
   for fold in spec['folds']:
    tr=np.array(fold['train']); te=np.array(fold['test']); m=ClassConditionalSoftPrompt(n_prompts=k,n_components=min(32,X.shape[1]),epochs=180,lr=.03,anchor=anchor,seed=42).fit(X[tr],y[tr]); p=m.predict_proba(X[te]); vals.append(metrics(y[te],p,len(classes)))
   row={'dataset':name,'variant':var}; row.update({f'{x}_mean':np.mean([v[x] for v in vals]) for x in vals[0]}); row.update({f'{x}_std':np.std([v[x] for v in vals],ddof=1) for x in vals[0]}); rows.append(row); print(row)
 pd.DataFrame(rows).to_csv(a.out,index=False)
