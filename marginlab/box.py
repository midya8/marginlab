"""Exact extrema of a declared affine function on a finite original-unit box.

Not a floating-point execution bound, plausible measurement set or medical result.
"""
from fractions import Fraction as F
import math,re

SCOPE='exact declared affine function over a finite rational box; not floating-point execution, real-world validity, clinical validation, probabilities, RBF or generalization'

def text(v):return str(v.numerator)+'/'+str(v.denominator)

def rat(s):
    if not isinstance(s,str) or len(s)>15000 or not re.fullmatch(r'-?(0|[1-9][0-9]*)/[1-9][0-9]*',s):raise ValueError('bounded canonical n/d strings required')
    a,b=s.split('/')
    if max(len(a.lstrip('-')),len(b))>4000:raise ValueError('oversized rational component')
    v=F(int(a),int(b))
    if text(v)!=s or max(v.numerator.bit_length(),v.denominator.bit_length())>24000:raise ValueError('noncanonical or oversized rational')
    return v

def binary(v):
    if isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v):raise ValueError('finite binary64 input required')
    return F.from_float(float(v))

def decimal(s):
    if not isinstance(s,str) or len(s)>70 or not re.fullmatch(r'-?(0|[1-9][0-9]*)(\.[0-9]{1,24})?',s):raise ValueError('bounded plain decimal string required')
    return F(s)

def affine(model,mean,scale):
    """Exact sum of saved alpha*y*x, then exact represented affine rescaling."""
    if not isinstance(model,dict) or model.get('kind')!='linear':raise ValueError('linear models only')
    x,y,alpha=model['x'],model['y'],model['alpha'];d=len(mean)
    if not 1<=d<=32 or len(scale)!=d or not 1<=len(x)<=1000 or len(y)!=len(x) or len(alpha)!=len(x) or any(len(row)!=d for row in x):raise ValueError('source dimensions differ')
    if any(type(t) is not int or t not in [-1,1] for t in y):raise ValueError('integer signed source labels required')
    mu=list(map(binary,mean));sd=list(map(binary,scale))
    if any(s<=0 for s in sd):raise ValueError('positive source scales required')
    aa=list(map(binary,alpha));xx=[list(map(binary,row)) for row in x]
    w=[sum((a*t*row[j] for a,t,row in zip(aa,y,xx)),F(0))/sd[j] for j in range(d)]
    c=binary(model['bias'])-sum((a*m for a,m in zip(w,mu)),F(0))
    return dict(weights=list(map(text,w)),bias=text(c))

def problem(aff,center,lower,upper):
    return dict(affine=aff,center=center,lower=lower,upper=upper)

def parse(p):
    if not isinstance(p,dict) or set(p)!={'affine','center','lower','upper'}:raise ValueError('problem fields differ')
    a=p['affine']
    if not isinstance(a,dict) or set(a)!={'weights','bias'}:raise ValueError('affine fields differ')
    arrays=[a['weights'],p['center'],p['lower'],p['upper']]
    if any(not isinstance(v,list) for v in arrays) or not 1<=len(arrays[0])<=32 or any(len(v)!=len(arrays[0]) for v in arrays):raise ValueError('box dimensions differ')
    w,x,lo,hi=[list(map(rat,v)) for v in arrays];b=rat(a['bias'])
    # Smaller per-input budgets bound arithmetic even for mutually coprime denominators.
    if any(max(q.numerator.bit_length(),q.denominator.bit_length())>512 for row in [w,x,lo,hi] for q in row) or max(b.numerator.bit_length(),b.denominator.bit_length())>4096:raise ValueError('input rational complexity exceeds budget')
    if any(l>v or v>h for l,v,h in zip(lo,x,hi)):raise ValueError('center must belong to every finite interval')
    return w,b,x,lo,hi

def produce(p):
    w,b,x,lo,hi=parse(p)
    low=[h if a<0 else l for a,l,h in zip(w,lo,hi)]
    high=[l if a<0 else h for a,l,h in zip(w,lo,hi)]
    score=lambda q:b+sum((a*v for a,v in zip(w,q)),F(0))
    mn,mx,center=score(low),score(high),score(x)
    labels=([-1] if mn<0 else [])+([1] if mx>=0 else [])
    proof=dict(version=1,problem=p,minimum=dict(point=list(map(text,low)),score=text(mn)),maximum=dict(point=list(map(text,high)),score=text(mx)),center_score=text(center),center_label=1 if center>=0 else -1,possible_labels=labels,stable=len(labels)==1)
    verify(proof);return proof

def verify(v):
    """Check all coordinate support inequalities and attained extrema, no optimizer."""
    if not isinstance(v,dict) or set(v)!={'version','problem','minimum','maximum','center_score','center_label','possible_labels','stable'} or type(v['version']) is not int or v['version']!=1:raise ValueError('proof schema differs')
    w,b,x,lo,hi=parse(v['problem']);values=[]
    for name,is_min in [('minimum',True),('maximum',False)]:
        witness=v[name]
        if not isinstance(witness,dict) or set(witness)!={'point','score'} or not isinstance(witness['point'],list) or len(witness['point'])!=len(w):raise ValueError('endpoint schema differs')
        point=list(map(rat,witness['point']));total=b
        for a,l,h,q in zip(w,lo,hi,point):
            if not l<=q<=h:raise ValueError('endpoint outside box')
            # Each endpoint must attain the lower/upper support of its own coordinate.
            candidates=(a*l,a*h);bound=min(candidates) if is_min else max(candidates)
            if a*q!=bound:raise ValueError('coordinate support not attained')
            total+=a*q
        if rat(witness['score'])!=total:raise ValueError('endpoint score differs')
        values.append(total)
    center=b+sum((a*q for a,q in zip(w,x)),F(0));labels=([-1] if values[0]<0 else [])+([1] if values[1]>=0 else [])
    if rat(v['center_score'])!=center or type(v['center_label']) is not int or v['center_label']!=(1 if center>=0 else -1):raise ValueError('center score or label differs')
    if not isinstance(v['possible_labels'],list) or any(type(t) is not int for t in v['possible_labels']) or v['possible_labels']!=labels or type(v['stable']) is not bool or v['stable']!=(len(labels)==1):raise ValueError('tie-aware label claim differs')
    return dict(valid=True,features=len(w),minimum=text(values[0]),maximum=text(values[1]),stable=v['stable'],possible_labels=labels,scope=SCOPE)

def from_report(report,model_id,request):
    if not isinstance(request,dict) or set(request)!={'center','lower','upper'}:raise ValueError('request needs only rational center, lower, upper arrays')
    candidate=next((q for q in report['candidates'] if q['id']==model_id),None)
    if candidate is None:raise ValueError('unknown model ID')
    a=affine(candidate['model'],report['mean'],report['scale'])
    return produce(problem(a,**request))
