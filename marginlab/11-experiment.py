"""Frozen protocol: select on validation; test opens only after selection."""
import json,math
from pathlib import Path
import numpy as np
from .core import fit,Model,kernel,audit_gram
from .dataset import load,split,FEATURES,HASHES
ARTIFACT=Path(__file__).parent/'artifacts/report.json'
CONFIGS=[(f'linear-{C:g}','linear',C,None) for C in (.01,.1,1.,10.)]+[(f'rbf-{C:g}','rbf',C,1/30) for C in (.1,1.,10.,100.)]

def metrics(y,scores):
    y=np.asarray(y);scores=np.asarray(scores);pred=np.where(scores>=0,1,-1);tp=int(np.sum((y==1)&(pred==1)));tn=int(np.sum((y==-1)&(pred==-1)));fp=int(np.sum((y==-1)&(pred==1)));fn=int(np.sum((y==1)&(pred==-1)))
    return dict(n=len(y),accuracy=float(np.mean(y==pred)),balanced_accuracy=.5*(tp/(tp+fn)+tn/(tn+fp)),hinge=float(np.mean(np.maximum(0,1-y*scores))),confusion=dict(tp=tp,tn=tn,fp=fp,fn=fn))

def run(raw=None,target=None,max_updates=50000):
    if raw is None:raw,target=load()
    raw=np.asarray(raw);target=np.asarray(target);train,val,test=split();mean=raw[train].mean(0);scale=raw[train].std(0)
    if np.any(scale<=0):raise ValueError('zero training scale')
    z=(raw-mean)/scale;candidates=[]
    for id,kind,C,gamma in CONFIGS:
        result=fit(z[train],target[train],C,kind,gamma,max_updates);model=Model.from_dict(result['model']);result.update(id=id,train=metrics(target[train],model.decision(z[train])),validation=metrics(target[val],model.decision(z[val])));candidates.append(result)
    # Fixed order breaks ties; hinge is not comparable to calibrated probability.
    selected=min(candidates,key=lambda a:(-a['validation']['balanced_accuracy'],a['validation']['hinge'],CONFIGS.index(next(c for c in CONFIGS if c[0]==a['id']))))['id']
    baseline={}
    # True nearest neighbor over all 30 standardized features; no weakened feature subset.
    for name,ids in [('train',train),('validation',val),('test',test)]:
        d=np.zeros((len(ids),len(train)))
        for j in range(30):d+=(z[ids,j,None]-z[train,j][None,:])**2
        scores=target[train][np.argmin(d,axis=1)];majority=1 if np.count_nonzero(target[train]==1)>=len(train)/2 else -1
        baseline[name]=dict(nearest_neighbor=metrics(target[ids],scores),majority=metrics(target[ids],np.full(len(ids),majority)))
    # Test scoring does not alter models or selection.
    for result in candidates:result['test']=metrics(target[test],Model.from_dict(result['model']).decision(z[test]))
    return dict(version=1,protocol=dict(seed=20261108,split_ids={k:v.tolist() for k,v in zip(('train','validation','test'),(train,val,test))},counts={k:len(v) for k,v in zip(('train','validation','test'),(train,val,test))},selection='maximize validation balanced accuracy, then minimize validation mean hinge, then fixed config order; no validation refit',max_updates=max_updates,tolerance=1e-5,C_normalization='P = 0.5 ||w||^2 + C * SUM hinge; not mean hinge'),source_hashes=HASHES,features=FEATURES,mean=mean.tolist(),scale=scale.tolist(),selected_id=selected,candidates=candidates,baselines=baseline,scope='historical public-data educational classifier; not clinical validation or calibrated probabilities')

def saved():return json.loads(ARTIFACT.read_text())

def audit(report):
    rows=[]
    for result in report['candidates']:
        m=Model.from_dict(result['model']);K=kernel(m.x,m.x,m.kind,m.gamma);a=audit_gram(K,m.y,m.alpha,m.C)
        for key in ('primal','dual','gap','kkt_residual','equality_residual','box_residual'):
            if not math.isclose(a[key],result['audit'][key],rel_tol=2e-7,abs_tol=2e-7):raise ValueError('final audit mismatch '+key)
        alpha=np.zeros(len(m.x));raw=np.zeros(len(m.x));previous=0.;max_eq=0.;max_box=0.;gain_total=0.
        for i,j,di,dj,eta,gain in result['pairs']:
            gradient=1-m.y*raw
            actual_eta=float(K[i,i]+K[j,j]-2*K[i,j])
            actual_gain=di*gradient[i]+dj*gradient[j]-.5*(di*di*K[i,i]+dj*dj*K[j,j]+2*di*dj*m.y[i]*m.y[j]*K[i,j])
            if not math.isclose(actual_eta,eta,rel_tol=1e-10,abs_tol=1e-10) or not math.isclose(actual_gain,gain,rel_tol=2e-7,abs_tol=1e-9):raise ValueError('incorrect pair curvature or gain')
            alpha[i]+=di;alpha[j]+=dj;raw+=di*m.y[i]*K[:,i]+dj*m.y[j]*K[:,j]
            max_eq=max(max_eq,abs(float(alpha@m.y)));max_box=max(max_box,0.,float(-alpha.min()),float(alpha.max()-m.C));gain_total+=gain
            if gain<=0 or abs(di*m.y[i]+dj*m.y[j])>1e-11:raise ValueError('bad pair witness')
        if result['updates']!=len(result['pairs']):raise ValueError('update count differs')
        if not np.allclose(alpha,m.alpha,rtol=0,atol=1e-10) or not math.isclose(gain_total,a['dual'],abs_tol=1e-6,rel_tol=1e-7):raise ValueError('pair replay differs')
        if max_box>1e-9 or max_eq>1e-8:raise ValueError('infeasible pair replay')
        if result['status']=='kkt_and_gap_tolerance' and not (a['pair_violation']<=result['tolerance']+1e-10 and a['gap']<=result['tolerance']*(1+abs(a['primal']))+1e-9):raise ValueError('false stop status')
        rows.append(dict(id=result['id'],updates=len(result['pairs']),status=result['status'],maximum_equality_residual=max_eq,maximum_box_residual=max_box,**{k:a[k] for k in ('primal','dual','gap','kkt_residual')}))
    return dict(selected_id=report['selected_id'],checks=rows,scope='floating-point numerical residuals and full pair replay, not exact-rational certificates or origin authentication')

if __name__=='__main__':
    r=run();ARTIFACT.write_text(json.dumps(r,separators=(',',':'),allow_nan=False)+'\n');print(r['selected_id'],r['protocol']['counts']);print([(a['id'],a['status'],a['validation']['balanced_accuracy'],a['test']['accuracy']) for a in r['candidates']]);print(audit(r))
