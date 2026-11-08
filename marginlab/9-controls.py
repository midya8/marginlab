"""Supplied mathematical controls, not causal ablations of the medical dataset."""
import numpy as np
from .core import fit,Model,audit_gram,kernel

def controls():
    x=np.array([[-2.],[-1.],[1.],[2.]]);y=np.array([-1,-1,1,1]);short=fit(x,y,max_updates=1);full=fit(x,y)
    duplicate_x=np.array([[-1.],[-1.],[1.],[1.]]);duplicate_y=np.array([-1,-1,1,1]);a=np.array([.5,0,.5,0]);b=np.array([0,.5,0,.5]);K=kernel(duplicate_x,duplicate_x)
    zero=fit([[0.],[0.]],[-1,1],C=.7)
    return dict(accuracy_not_optimization=dict(short=short,full=full,short_accuracy=float(np.mean(Model.from_dict(short['model']).predict(x)==y))),duplicate_support=dict(x=duplicate_x.tolist(),y=duplicate_y.tolist(),alpha_a=a.tolist(),alpha_b=b.tolist(),audit_a=audit_gram(K,duplicate_y,a,1.),audit_b=audit_gram(K,duplicate_y,b,1.),decision_grid=[-3,-1,0,1,3]),zero_curvature=zero)
