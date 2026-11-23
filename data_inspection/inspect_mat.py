import argparse, json, os
from pathlib import Path
import numpy as np
from scipy.io import loadmat

def inspect(path):
 d=loadmat(path); out={'file':Path(path).name,'variables':{}}
 for k,v in d.items():
  if k.startswith('__'): continue
  a=np.asarray(v)
  flat=a.reshape(-1)
  out['variables'][k]={'shape':list(a.shape),'dtype':str(a.dtype),'n':int(a.shape[0]),'nan':int(np.isnan(a).sum()),'inf':int(np.isinf(a).sum()),'unique_rows':None}
  if a.ndim>=2:
   try: out['variables'][k]['unique_rows']=int(np.unique(a.reshape(a.shape[0],-1),axis=0).shape[0])
   except Exception: pass
 return out
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='data'); ap.add_argument('--out-dir',default='results/inspection'); args=ap.parse_args(); Path(args.out_dir).mkdir(parents=True,exist_ok=True)
 allout=[inspect(p) for p in sorted(Path(args.data_dir).glob('*.mat'))]
 json.dump(allout,open(Path(args.out_dir)/'inspection.json','w'),indent=2)
 for x in allout: print(x['file'],x['variables'])
