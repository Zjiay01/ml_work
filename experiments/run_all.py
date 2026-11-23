import argparse,json,sys,time,traceback
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV,StratifiedKFold
from sklearn.metrics import roc_auc_score,accuracy_score,f1_score,precision_score,recall_score
sys.path.insert(0,str(Path(__file__).resolve().parents[1])); from preprocessing.datasets import DATASETS,load_dataset
from methods.soft_prompt import ClassConditionalSoftPrompt

def auc(y,p,C):
 try: return roc_auc_score(y,p if C>2 else p[:,1],multi_class='ovr' if C>2 else 'raise',average='macro')
 except: return np.nan

def metrics(y,p,C):
 pred=p.argmax(1); return {'auc':auc(y,p,C),'macro_f1':f1_score(y,pred,average='macro'),'accuracy':accuracy_score(y,pred),'macro_precision':precision_score(y,pred,average='macro',zero_division=0),'macro_recall':recall_score(y,pred,average='macro',zero_division=0)}

def baseline_specs(name,n):
 fmr='fMRI' in name
 pca_dim=[16,32,64] if fmr else [None]
 return {
 'SVM':(Pipeline([('imp',SimpleImputer()),('sc',StandardScaler()),('pca',PCA(random_state=1)),('clf',SVC(probability=True,class_weight='balanced',random_state=1))]), {'pca__n_components':pca_dim,'pca__whiten':[True] if fmr else [False],'clf__kernel':['linear','rbf'],'clf__C':[0.1,1,10],'clf__gamma':['scale',0.01]}),
 'MLP':(Pipeline([('imp',SimpleImputer()),('sc',StandardScaler()),('sel',SelectKBest(f_classif,k=min(64,n))),('clf',MLPClassifier(max_iter=500,early_stopping=True,random_state=1))]), {'sel__k':[min(32,n),min(64,n)],'clf__hidden_layer_sizes':[(32,),(64,)],'clf__alpha':[0.001,0.01,0.1]}),
 }

def fit_baseline(kind,X,y,tr,te,name):
 pipe,grid=baseline_specs(name,X.shape[1])[kind]; cv=StratifiedKFold(3,shuffle=True,random_state=123); score='roc_auc_ovr' if len(np.unique(y))>2 else 'roc_auc'; gs=GridSearchCV(pipe,grid,cv=cv,scoring=score,n_jobs=-1,refit=True); gs.fit(X[tr],y[tr]); return gs.predict_proba(X[te]),str(gs.best_params_)

def run_prompt(X,y,tr,te):
 # Conservative fold-local model selection on training folds only.
 m=ClassConditionalSoftPrompt(n_prompts=4,n_components=min(32,X.shape[1]),epochs=180,lr=0.03,anchor=0.05,seed=42); m.fit(X[tr],y[tr]); return m.predict_proba(X[te]),{'n_prompts':4,'n_components':m.pca.n_components_,'train_loss':m.loss_}
if __name__=='__main__':
 ap=argparse.ArgumentParser(); ap.add_argument('--data-dir',default='../data'); ap.add_argument('--split-dir',default='../splits'); ap.add_argument('--out-dir',default='../results'); ap.add_argument('--log-dir',default='../logs'); args=ap.parse_args(); Path(args.out_dir).mkdir(parents=True,exist_ok=True); Path(args.log_dir).mkdir(parents=True,exist_ok=True)
 rows=[]; exp=[]
 for name in DATASETS:
  X,y,classes,_=load_dataset(name,args.data_dir); spec=json.load(open(Path(args.split_dir)/(name+'.json')))
  for method in ['SVM','MLP','CCSP']:
   for fi,fold in enumerate(spec['folds']):
    t=time.time(); tr=np.array(fold['train']); te=np.array(fold['test'])
    try:
     if method=='CCSP': p,pars=run_prompt(X,y,tr,te)
     else: p,pars=fit_baseline(method,X,y,tr,te,name)
     mm=metrics(y[te],p,len(classes)); mm.update(dataset=name,method=method,fold=fi,seconds=time.time()-t,params=pars); rows.append(mm)
     np.savez(Path(args.out_dir)/f'{name}_{method}_fold{fi}.npz',y=y[te],proba=p,test_idx=te)
     print(name,method,fi,mm)
    except Exception as e:
     exp.append({'dataset':name,'method':method,'fold':fi,'status':'FAILED','error':repr(e),'traceback':traceback.format_exc()}); print('FAILED',name,method,fi,e)
  
 # CatBoost optional, kept separate so missing package is explicit.
 try:
  from catboost import CatBoostClassifier
  for name in DATASETS:
   X,y,classes,_=load_dataset(name,args.data_dir); spec=json.load(open(Path(args.split_dir)/(name+'.json')))
   for fi,fold in enumerate(spec['folds']):
    tr=np.array(fold['train']); te=np.array(fold['test']); t=time.time(); clf=CatBoostClassifier(iterations=250,depth=4,learning_rate=.03,verbose=False,random_seed=1,loss_function='MultiClass' if len(classes)>2 else 'Logloss',auto_class_weights='Balanced'); clf.fit(X[tr],y[tr]); p=clf.predict_proba(X[te]); mm=metrics(y[te],p,len(classes)); mm.update(dataset=name,method='CatBoost',fold=fi,seconds=time.time()-t,params='fixed course baseline'); rows.append(mm); np.savez(Path(args.out_dir)/f'{name}_CatBoost_fold{fi}.npz',y=y[te],proba=p,test_idx=te)
 except Exception as e: exp.append({'status':'FAILED','method':'CatBoost','error':repr(e),'traceback':traceback.format_exc()})
 pd.DataFrame(rows).to_csv(Path(args.out_dir)/'fold_metrics.csv',index=False); pd.DataFrame(exp).to_json(Path(args.log_dir)/'experiment_log.json',orient='records',indent=2)
