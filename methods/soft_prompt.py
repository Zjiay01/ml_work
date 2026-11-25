"""NumPy implementation of class-conditional soft prompt tuning."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

class ClassConditionalSoftPrompt:
 def __init__(self,n_prompts=4,n_components=32,epochs=250,lr=0.03,anchor=0.05,seed=42): self.n_prompts=n_prompts; self.n_components=n_components; self.epochs=epochs; self.lr=lr; self.anchor=anchor; self.seed=seed
 def fit(self,X,y):
  rng=np.random.default_rng(self.seed); self.scaler=StandardScaler(); Z=self.scaler.fit_transform(X); nc=max(2,min(self.n_components,Z.shape[0]-1,Z.shape[1])); self.pca=PCA(n_components=nc,random_state=self.seed); H=self.pca.fit_transform(Z).astype('float64'); self.mean_=H.mean(0); self.std_=H.std(0)+1e-6; H=(H-self.mean_)/self.std_
  self.classes_=np.unique(y); C=len(self.classes_); D=H.shape[1]; prot=np.stack([H[y==c].mean(0) for c in self.classes_]); prot/=np.linalg.norm(prot,axis=1,keepdims=True)+1e-8; P=np.repeat(prot[:,None,:],self.n_prompts,axis=1)+.02*rng.normal(size=(C,self.n_prompts,D)); b=np.zeros(C); logt=0.; counts=np.bincount(y,minlength=C); w=len(y)/(C*np.maximum(counts,1)); self.loss_=0
  hn=H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-8)
  for _ in range(self.epochs):
   pn=P/(np.linalg.norm(P,axis=2,keepdims=True)+1e-8); sims=np.einsum('nd,ckd->nck',hn,pn).mean(2); logits=sims*np.exp(logt)+b; logits-=logits.max(1,keepdims=True); E=np.exp(logits); probs=E/E.sum(1,keepdims=True); loss=-np.mean(w[y]*np.log(probs[np.arange(len(y)),y]+1e-12))+self.anchor*np.mean((P-prot[:,None,:])**2); self.loss_=float(loss)
   dlog=probs; dlog[np.arange(len(y)),y]-=1; dlog*=w[y,None]/len(y); db=dlog.sum(0); dlt=np.sum(dlog*sims*np.exp(logt)); ds=dlog*np.exp(logt)
   # gradient through class mean similarities and normalized prompts
   gradP=np.zeros_like(P)
   for c in range(C):
    g=np.zeros((self.n_prompts,D))
    for k in range(self.n_prompts):
     v=P[c,k]; nv=np.linalg.norm(v)+1e-8; u=v/nv; q=ds[:,c]/self.n_prompts; grad_u=(q[:,None]*hn).sum(0); g[k]=(grad_u-u*np.dot(u,grad_u))/nv
    gradP[c]=g
   P-=self.lr*(gradP+2*self.anchor*(P-prot[:,None,:])); b-=self.lr*db; logt-=self.lr*dlt
  self.prompts_=P; self.bias_=b; self.temp_=float(np.exp(logt)); return self
 def _z(self,X): return (self.pca.transform(self.scaler.transform(X))-self.mean_)/self.std_
 def predict_proba(self,X):
  H=self._z(X); P=self.prompts_; hn=H/(np.linalg.norm(H,axis=1,keepdims=True)+1e-8); pn=P/(np.linalg.norm(P,axis=2,keepdims=True)+1e-8); s=np.einsum('nd,ckd->nck',hn,pn).mean(2)*self.temp_+self.bias_; s-=s.max(1,keepdims=True); e=np.exp(s); return e/e.sum(1,keepdims=True)
