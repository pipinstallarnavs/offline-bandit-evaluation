import argparse,json,urllib.request
from pathlib import Path
import numpy as np
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
ROOT=Path(__file__).parent
def policy(X):
 return LogisticRegression(max_iter=300,multi_class='auto').fit(X,np.argmax(X.reshape(len(X),-1),axis=1)%10)
def run(seed=7,n=5000):
 d=load_digits(); X=d.data/16.; labels=d.target; rng=np.random.default_rng(seed); Xtr,Xte,ytr,yte=train_test_split(X,labels,test_size=.4,random_state=seed,stratify=labels); clf=LogisticRegression(max_iter=300).fit(Xtr,ytr); probs=clf.predict_proba(Xtr); actions=rng.choice(10,len(Xtr),p=np.ones(10)/10) # uniform logging has overlap
 reward=(actions==ytr).astype(float); target=clf.predict(Xte); truth=float(np.mean(target==yte));
 # target policy is classifier; behavior propensity is known 0.1.
 target_actions=clf.predict(Xtr); ratio=(target_actions==actions).astype(float)/.1; ips=float(np.mean(ratio*reward)); snips=float(np.sum(ratio*reward)/max(np.sum(ratio),1e-9))
 # Outcome model: class-conditional reward from training logs, with ridge-like smoothing.
 q=np.full(10,.5)
 for a in range(10):
  seen=actions==a; q[a]=(reward[seen].sum()+1)/(seen.sum()+2)
 dr=float(np.mean(q[target_actions]+ratio*(reward-q[actions])))
 return {'n_train':len(Xtr),'n_test':len(Xte),'true_policy_accuracy':truth,'direct_logged_reward':float(reward.mean()),'ips':ips,'snips':snips,'doubly_robust':dr,'seed':seed,'behavior_policy':'uniform p(a|x)=0.1'}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--seed',type=int,default=7);a=ap.parse_args();r=run(a.seed);(ROOT/'results').mkdir(exist_ok=True);(ROOT/'results/report.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
if __name__=='__main__':main()
