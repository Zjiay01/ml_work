from pathlib import Path
import numpy as np
from scipy.io import loadmat
from sklearn.preprocessing import LabelEncoder

def fisher_connectivity(x):
 # x: n x regions x time. Correlation is computed per subject, then Fisher-z transformed.
 out=[]; iu=np.triu_indices(x.shape[1],1)
 for subj in x:
  c=np.corrcoef(subj)
  c=np.clip(c,-0.999999,0.999999)
  out.append(np.arctanh(c[iu]))
 return np.asarray(out,dtype=np.float32)

def load_dataset(name,data_dir='data'):
 p=Path(data_dir)
 if name=='ADNI':
  d=loadmat(p/'ADNI(1).mat'); keys=['AD','MCIn','MCIp','NC']; X=np.concatenate([d[k] for k in keys]); y=np.concatenate([[k]*len(d[k]) for k in keys]); groups=np.array([0]*93+[1]*93)
 elif name=='ADNI_fMRI':
  d=loadmat(p/'ADNI_90_120_fMRI(1).mat'); keys=['AD','EMCI','LMCI','NC']; X=fisher_connectivity(np.concatenate([d[k] for k in keys])); y=np.concatenate([[k]*len(d[k]) for k in keys]); groups=None
 elif name=='FTD_fMRI':
  d=loadmat(p/'FTD_90_200_fMRI(1).mat'); keys=['FTD','NC']; X=fisher_connectivity(np.concatenate([d[k] for k in keys])); y=np.concatenate([[k]*len(d[k]) for k in keys]); groups=None
 elif name=='OCD_fMRI':
  d=loadmat(p/'OCD_90_200_fMRI(1).mat'); keys=['NC','OCD']; raw=np.concatenate([d[k] for k in keys]); y=np.concatenate([[k]*len(d[k]) for k in keys]);
  flat=raw.reshape(raw.shape[0],-1); _,keep=np.unique(flat,axis=0,return_index=True); keep=np.sort(keep); raw,y=raw[keep],y[keep]; X=fisher_connectivity(raw); groups=None
 elif name=='PPMI':
  d=loadmat(p/'PPMI(1).mat'); keys=['NC','PD']; X=np.concatenate([d[k] for k in keys]).astype(np.float32); y=np.concatenate([[k]*len(d[k]) for k in keys]); groups=np.repeat(np.arange(3),98)
 else: raise ValueError(name)
 le=LabelEncoder(); yi=le.fit_transform(y)
 return X.astype(np.float32),yi,le.classes_.tolist(),groups

DATASETS=['ADNI','ADNI_fMRI','FTD_fMRI','OCD_fMRI','PPMI']
