import argparse,json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from datasets import DATASETS,load_dataset
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='../data'); ap.add_argument('--out-dir',default='../splits'); args=ap.parse_args(); Path(args.out_dir).mkdir(parents=True,exist_ok=True)
 for name in DATASETS:
  X,y,classes,g=load_dataset(name,args.data_dir); sk=StratifiedKFold(5,shuffle=True,random_state=42); folds=[]
  for tr,te in sk.split(X,y): folds.append({'train':tr.tolist(),'test':te.tolist()})
  json.dump({'dataset':name,'n':len(y),'d':X.shape[1],'classes':classes,'folds':folds},open(Path(args.out_dir)/(name+'.json'),'w'),indent=2)
  print(name,X.shape,dict(zip(classes,[int((y==i).sum()) for i in range(len(classes))])))
