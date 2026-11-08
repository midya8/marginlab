"""Two-coordinate soft-margin dual ascent. NumPy supplies arrays, not an SVM."""
from dataclasses import dataclass
import math
import numpy as np


def matrix(value):
    a=np.asarray(value,dtype=float)
    if a.ndim!=2 or min(a.shape)==0 or not np.isfinite(a).all():raise ValueError('expected a finite nonempty matrix')
    return a


def labels(value,n):
    a=np.asarray(value,dtype=float)
    if a.shape!=(n,) or not np.isin(a,[-1,1]).all() or len(np.unique(a))!=2:raise ValueError('expected both labels -1 and +1')
    return a


def kernel(a,b,kind='linear',gamma=None):
    a=matrix(a);b=matrix(b)
    if a.shape[1]!=b.shape[1]:raise ValueError('feature dimensions differ')
    if kind=='linear':return a@b.T
    if kind!='rbf' or gamma is None or not math.isfinite(gamma) or gamma<=0:raise ValueError('invalid kernel or gamma')
    # Distances by explicit differences avoid cancellation at duplicate points.
    distance=np.zeros((len(a),len(b)))
    for j in range(a.shape[1]):distance+=(a[:,j,None]-b[None,:,j])**2
    return np.exp(-gamma*distance)


def eligible(alpha,y,C):
    return ((y==1)&(alpha<C))|((y==-1)&(alpha>0)),((y==1)&(alpha>0))|((y==-1)&(alpha<C))


def audit_gram(K,y,alpha,C):
    K=matrix(K);y=labels(y,len(K));a=np.asarray(alpha,dtype=float)
    if K.shape!=(len(y),len(y)) or a.shape!=y.shape or not np.isfinite(a).all() or not math.isfinite(C) or C<=0:raise ValueError('invalid dual state')
    raw=K@(a*y);h=y-raw;up,low=eligible(a,y,C)
    upper=float(np.max(h[up]));lower=float(np.min(h[low]));bias=(upper+lower)/2
    margins=y*(raw+bias);norm=float((a*y)@raw);dual=float(a.sum()-.5*norm);hinge=np.maximum(0,1-margins);primal=float(.5*norm+C*hinge.sum())
    residual=np.maximum(np.where(a<C,1-margins,0),np.where(a>0,margins-1,0))
    return dict(bias=bias,primal=primal,dual=dual,gap=primal-dual,pair_violation=upper-lower,kkt_residual=float(np.max(residual)),equality_residual=abs(float(y@a)),box_residual=max(0.,float(-a.min()),float(a.max()-C)),norm_squared=norm,margins=margins.tolist(),hinge=hinge.tolist(),support_count=int(np.count_nonzero(a>0)),free_count=int(np.count_nonzero((a>0)&(a<C))),bias_interval=[upper,lower])


@dataclass
class Model:
    x:np.ndarray
    y:np.ndarray
    alpha:np.ndarray
    bias:float
    C:float
    kind:str='linear'
    gamma:float|None=None

    def __post_init__(self):
        self.x=matrix(self.x);self.y=labels(self.y,len(self.x));self.alpha=np.asarray(self.alpha,dtype=float)
        if self.alpha.shape!=self.y.shape or not np.isfinite(self.alpha).all() or not math.isfinite(self.bias) or not math.isfinite(self.C) or self.C<=0:raise ValueError('invalid model')
        if np.min(self.alpha)<-1e-10 or np.max(self.alpha)>self.C+1e-10 or abs(float(self.alpha@self.y))>1e-7:raise ValueError('infeasible dual model')
        kernel(self.x[:1],self.x[:1],self.kind,self.gamma)

    def decision(self,x):return kernel(matrix(x),self.x,self.kind,self.gamma)@(self.alpha*self.y)+self.bias
    def predict(self,x):return np.where(self.decision(x)>=0,1,-1)
    def to_dict(self):return dict(x=self.x.tolist(),y=self.y.astype(int).tolist(),alpha=self.alpha.tolist(),bias=self.bias,C=self.C,kind=self.kind,gamma=self.gamma)
    @classmethod
    def from_dict(cls,value):return cls(**value)


def fit(x,y,C=1.,kind='linear',gamma=None,max_updates=50000,tolerance=1e-5):
    x=matrix(x);y=labels(y,len(x))
    if not math.isfinite(C) or C<=0 or isinstance(max_updates,bool) or not isinstance(max_updates,int) or not 0<=max_updates<=1000000 or not math.isfinite(tolerance) or tolerance<=0:raise ValueError('invalid fit controls')
    K=kernel(x,x,kind,gamma);alpha=np.zeros(len(x));raw=np.zeros(len(x));pairs=[];history=[];status='budget_exhausted'
    for iteration in range(max_updates+1):
        up,low=eligible(alpha,y,C);h=y-raw;i=int(np.argmax(np.where(up,h,-np.inf)));j=int(np.argmin(np.where(low,h,np.inf)));violation=float(h[i]-h[j])
        if iteration%100==0 or iteration==max_updates or violation<=tolerance:
            raw=K@(alpha*y);a=audit_gram(K,y,alpha,C);history.append(dict(update=iteration,**{k:v for k,v in a.items() if k not in ('margins','hinge')}))
            if a['pair_violation']<=tolerance and a['gap']<=tolerance*(1+abs(a['primal'])):
                status='kkt_and_gap_tolerance';break
            # Re-select from refreshed scores before using a cached gap.
            h=y-raw;i=int(np.argmax(np.where(up,h,-np.inf)));j=int(np.argmin(np.where(low,h,np.inf)));violation=float(h[i]-h[j])
        if iteration==max_updates:break
        limit=min(C-alpha[i] if y[i]==1 else alpha[i],alpha[j] if y[j]==1 else C-alpha[j])
        eta=float(K[i,i]+K[j,j]-2*K[i,j])
        if eta < -1e-10:raise ValueError('negative pair curvature for supported PSD kernel')
        step=min(limit,violation/eta) if eta>0 else (limit if violation>0 else 0.)
        if i==j or step<=0 or not math.isfinite(step):status='no_progress';break
        gain=step*violation-.5*eta*step*step
        if gain<=0:status='no_progress';break
        oldi=float(alpha[i]);oldj=float(alpha[j]);alpha[i]+=step*y[i];alpha[j]-=step*y[j]
        # Force exact box endpoints through the limiting variable, then derive its partner.
        if step==limit:
            if limit==(C-oldi if y[i]==1 else oldi):alpha[i]=C if y[i]==1 else 0.;alpha[j]=oldj-y[j]*y[i]*(alpha[i]-oldi)
            else:alpha[j]=0. if y[j]==1 else C;alpha[i]=oldi-y[i]*y[j]*(alpha[j]-oldj)
        if alpha[i]<-1e-10 or alpha[j]<-1e-10 or alpha[i]>C+1e-10 or alpha[j]>C+1e-10:raise ArithmeticError('dual box drift')
        raw+=y[i]*(alpha[i]-oldi)*K[:,i]+y[j]*(alpha[j]-oldj)*K[:,j]
        pairs.append([i,j,float(alpha[i]-oldi),float(alpha[j]-oldj),float(eta),float(gain)])
    a=audit_gram(K,y,alpha,C);updates=len(pairs)
    if not history or history[-1]['update']!=updates:history.append(dict(update=updates,**{k:v for k,v in a.items() if k not in ('margins','hinge')}))
    model=Model(x,y,alpha,a['bias'],C,kind,gamma)
    return dict(model=model.to_dict(),status=status,updates=updates,budget=max_updates,tolerance=tolerance,audit=a,history=history,pairs=pairs)
