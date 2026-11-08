from marginlab.core import fit
from marginlab.dataset import load,split
import numpy as np,time
x,y=load();train,val,test=split();mean=x[train].mean(0);scale=x[train].std(0);z=(x-mean)/scale
for kind,C in [('linear',v) for v in (.01,.1,1,10)]+[('rbf',v) for v in (.1,1,10,100)]:
 t=time.time();a=fit(z[train],y[train],C,kind,1/30 if kind=='rbf' else None);print(kind,C,a['status'],a['updates'],{k:a['audit'][k] for k in ('gap','kkt_residual','equality_residual','box_residual')},time.time()-t,flush=True)
