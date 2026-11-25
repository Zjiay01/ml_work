import sys,json,argparse
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,GridSearchCV
from sklearn.metrics import roc_auc_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1])); from preprocessing.datasets import DATASETS,load_dataset,fisher_connectivity
from scipy.io import loadmat

def loadrep(name,dd):
 X,y,cl,_=load_dataset(name,dd); reps={'raw':X}
 if 'fMRI' in name:
  fn={'ADNI_fMRI':'ADNI_90_120_fMRI(1).mat','FTD_fMRI':'FTD_90_200_fMRI(1).mat','OCD_fMRI':'OCD_90_200_fMRI(1).mat'}[name]; keys={'ADNI_fMRI':['AD','EMCI','LMCI','NC'],'FTD_fMRI':['FTD','NC'],'OCD_fMRI':['NC','OCD']}[name]; d=loadmat(Path(dd)/fn); raw=np.concatenate([d[k] for k in keys]);
  if name=='OCD_fMRI': flat=raw.reshape(len(raw),-1); _,keep=np.unique(flat,axis=0,return_index=True); raw=raw[np.sort(keep)]
  conn=fisher_connectivity(raw); tm=raw.mean(2); ts=raw.std(2); iu=np.triu_indices(90,1); st=[]
  for row in conn:
   mat=np.zeros((90,90)); mat[iu]=row; mat=mat+mat.T; st.append(mat.mean(1))
  reps['summary']=np.c_[tm,ts,np.array(st)]
  reps['summary_conn']=np.c_[tm,ts,np.array(st),conn]
 return reps,y,cl

def auc(y,p,c): return roc_auc_score(y,p[:,1] if c==2 else p,multi_class='ovr' if c>2 else 'raise',average='macro')
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='../data'); ap.add_argument('--split-dir',default='../splits'); ap.add_argument('--out',default='../results/explore_quick.csv'); a=ap.parse_args(); rows=[]
 for name in DATASETS:
  reps,y,cl=loadrep(name,a.data_dir); sp=json.load(open(Path(a.split_dir)/(name+'.json')))
  for rep,X in reps.items():
   for pca in ([8,16,32,64] if 'fMRI' in name else [None]):
    vals=[]
    for f in sp['folds']:
     tr=np.array(f['train']); te=np.array(f['test']); pipe=Pipeline([('sc',StandardScaler()),('pca',PCA(n_components=pca,random_state=1)),('clf',LogisticRegression(max_iter=1000,class_weight='balanced',C=1))]); pipe.fit(X[tr],y[tr]); vals.append(auc(y[te],pipe.predict_proba(X[te]),len(cl)))
    rows.append({'dataset':name,'rep':rep,'pca':pca,'auc':np.mean(vals),'std':np.std(vals,ddof=1)}); print(rows[-1])
 pd.DataFrame(rows).to_csv(a.out,index=False)
