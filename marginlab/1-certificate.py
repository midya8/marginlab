"""Exact weak-duality bounds for a declared finite linear SVM problem.

No raw-real-standardization, RBF, generalization or provenance certificate.
"""
from fractions import Fraction as F
import math


def exact(value):
    if isinstance(value,bool) or not isinstance(value,(float,int)) or not math.isfinite(value):
        raise ValueError('finite numeric binary64 value required')
    return F.from_float(float(value))


def text(value):
    return str(value.numerator)+'/'+str(value.denominator)


def rational(value):
    if not isinstance(value,str) or len(value)>2500 or value.count('/')!=1:
        raise ValueError('bounded canonical rational required')
    a,b=value.split('/')
    try:
        out=F(int(a),int(b))
    except (ValueError,ZeroDivisionError) as e:
        raise ValueError('invalid rational') from e
    if text(out)!=value or out.numerator.bit_length()>4096 or out.denominator.bit_length()>4096:
        raise ValueError('noncanonical or oversized rational')
    return out


def dot(a,b):
    return sum((x*y for x,y in zip(a,b)),F(0))


def inputs(problem):
    if not isinstance(problem,dict) or set(problem)!={'x','y','C'}:
        raise ValueError('problem fields differ')
    raw,y=problem['x'],problem['y']
    if not isinstance(raw,list) or not 1<=len(raw)<=1000 or not isinstance(y,list) or len(y)!=len(raw):
        raise ValueError('problem rows differ')
    d=len(raw[0]) if isinstance(raw[0],list) else 0
    if not 1<=d<=100 or any(not isinstance(row,list) or len(row)!=d for row in raw):
        raise ValueError('problem columns differ')
    if any(type(label) is not int or label not in (-1,1) for label in y):
        raise ValueError('labels must be integer +/-1')
    x=[[exact(v) for v in row] for row in raw];C=exact(problem['C'])
    if C<=0:raise ValueError('C must be positive')
    return x,y,C


def produce(model):
    if model.get('kind')!='linear':raise ValueError('exact certificate supports linear kernels only')
    problem={k:model[k] for k in ('x','y','C')};x,y,C=inputs(problem)
    if not isinstance(model.get('alpha'),list) or len(model['alpha'])!=len(x):raise ValueError('alpha rows differ')
    original=[exact(v) for v in model['alpha']];b=exact(model['bias'])
    primal_w=[sum((a*label*row[j] for a,label,row in zip(original,y,x)),F(0)) for j in range(len(x[0]))]
    repaired=[max(F(0),min(C,a)) for a in original]
    excess=dot(repaired,y);label=1 if excess>=0 else -1;remaining=abs(excess)
    for i in range(len(repaired)):
        if y[i]==label:
            change=min(repaired[i],remaining);repaired[i]-=change;remaining-=change
    if remaining:raise ArithmeticError('insufficient repair mass')
    slacks=[max(F(0),1-label*(dot(primal_w,row)+b)) for label,row in zip(y,x)]
    dual_w=[sum((a*label*row[j] for a,label,row in zip(repaired,y,x)),F(0)) for j in range(len(x[0]))]
    primal=dot(primal_w,primal_w)/2+C*sum(slacks,F(0));dual=sum(repaired,F(0))-dot(dual_w,dual_w)/2
    proof=dict(version=1,problem=problem,primal=dict(w=list(map(text,primal_w)),bias=text(b),slacks=list(map(text,slacks))),dual=dict(alpha=list(map(text,repaired))),objectives=dict(primal=text(primal),dual=text(dual),gap=text(primal-dual)),repair=dict(original_alpha=list(map(text,original)),original_equality=text(dot(original,y)),original_box=text(max([F(0)]+[-a for a in original]+[a-C for a in original])),clipped_equality=text(excess),l1_change=text(sum((abs(a-b) for a,b in zip(original,repaired)),F(0)))))
    verify(proof)
    return proof


def verify(proof):
    """Recompute primal feasibility, dual feasibility and bounds, without fitting."""
    if not isinstance(proof,dict) or set(proof)!={'version','problem','primal','dual','objectives','repair'} or type(proof['version']) is not int or proof['version']!=1:
        raise ValueError('certificate schema differs')
    x,y,C=inputs(proof['problem']);p=proof['primal'];d=proof['dual'];o=proof['objectives'];r=proof['repair']
    if not all(isinstance(v,dict) for v in [p,d,o,r]):raise ValueError('witness objects required')
    if set(p)!={'w','bias','slacks'} or set(d)!={'alpha'} or set(o)!={'primal','dual','gap'} or set(r)!={'original_alpha','original_equality','original_box','clipped_equality','l1_change'}:
        raise ValueError('certificate fields differ')
    if not all(isinstance(v,list) for v in [p['w'],p['slacks'],d['alpha'],r['original_alpha']]):raise ValueError('vectors required')
    if len(p['w'])!=len(x[0]) or any(len(v)!=len(x) for v in [p['slacks'],d['alpha'],r['original_alpha']]):raise ValueError('witness dimensions differ')
    w=list(map(rational,p['w']));b=rational(p['bias']);slacks=list(map(rational,p['slacks']));alpha=list(map(rational,d['alpha']));original=list(map(rational,r['original_alpha']))
    if any(a<0 or a>C for a in alpha) or dot(alpha,y)!=0:raise ValueError('dual infeasible')
    if any(s<0 or label*(dot(w,row)+b)+s<1 for s,label,row in zip(slacks,y,x)):raise ValueError('primal infeasible')
    # The primal vector is deliberately independent of repaired dual feasibility.
    expected_w=[sum((a*label*row[j] for a,label,row in zip(original,y,x)),F(0)) for j in range(len(w))]
    if w!=expected_w:raise ValueError('original primal weights differ')
    dw=[sum((a*label*row[j] for a,label,row in zip(alpha,y,x)),F(0)) for j in range(len(w))]
    P=dot(w,w)/2+C*sum(slacks,F(0));D=sum(alpha,F(0))-dot(dw,dw)/2
    if P<D or any(rational(o[k])!=v for k,v in [('primal',P),('dual',D),('gap',P-D)]):raise ValueError('objectives differ')
    clipping=[max(F(0),min(C,a)) for a in original]
    for key,value in [('original_equality',dot(original,y)),('original_box',max([F(0)]+[-a for a in original]+[a-C for a in original])),('clipped_equality',dot(clipping,y)),('l1_change',sum((abs(a-b) for a,b in zip(original,alpha)),F(0)))]:
        if rational(r[key])!=value:raise ValueError('repair accounting differs')
    if any(a>clipped for a,clipped in zip(alpha,clipping)):raise ValueError('repair is not a reduction')
    return dict(valid=True,rows=len(x),features=len(w),exact_primal=text(P),exact_dual=text(D),exact_gap=text(P-D),scope='exact weak-duality bound for declared represented standardized linear problem only; not source authentication, RBF, generalization or clinical validation')
