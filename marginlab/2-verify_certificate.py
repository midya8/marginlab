"""Independent proof reader. No optimizer or certificate-producer import."""
import math
from fractions import Fraction


def check(document):
    def require(condition,message):
        if not condition:raise ValueError(message)
    def fraction(value):
        require(type(value) is str and len(value)<=2500 and value.count('/')==1,'rational syntax')
        try:
            n,d=value.split('/');q=Fraction(int(n),int(d))
        except (ValueError,ZeroDivisionError) as e:raise ValueError('rational syntax') from e
        require(value==str(q.numerator)+'/'+str(q.denominator) and max(q.numerator.bit_length(),q.denominator.bit_length())<=4096,'canonical bounded rational')
        return q
    def binary(value):
        require(type(value) in (int,float) and math.isfinite(value),'finite numeric problem value')
        return Fraction(*float(value).as_integer_ratio())
    def vector(values,n):
        require(type(values) is list and len(values)==n,'vector dimension')
        return [fraction(v) for v in values]
    def inner(a,b):return sum([u*v for u,v in zip(a,b)],Fraction(0))
    require(type(document) is dict and set(document)=={'version','problem','primal','dual','objectives','repair'},'top schema')
    require(type(document['version']) is int and document['version']==1,'version')
    data=document['problem'];require(type(data) is dict and set(data)=={'x','y','C'},'problem schema')
    require(type(data['x']) is list and 1<=len(data['x'])<=1000,'row limit');n=len(data['x'])
    require(type(data['x'][0]) is list and 1<=len(data['x'][0])<=100,'feature limit');d=len(data['x'][0])
    require(all(type(row) is list and len(row)==d for row in data['x']),'feature dimensions')
    X=[[binary(v) for v in row] for row in data['x']];Y=data['y'];C=binary(data['C'])
    require(type(Y) is list and len(Y)==n and all(type(v) is int and v in (-1,1) for v in Y) and C>0,'labels/C')
    p=document['primal'];a=document['dual'];o=document['objectives'];r=document['repair']
    require(all(type(v) is dict for v in [p,a,o,r]),'object witnesses')
    require(set(p)=={'w','bias','slacks'} and set(a)=={'alpha'} and set(o)=={'primal','dual','gap'} and set(r)=={'original_alpha','original_equality','original_box','clipped_equality','l1_change'},'witness schema')
    W=vector(p['w'],d);B=fraction(p['bias']);S=vector(p['slacks'],n);A=vector(a['alpha'],n);O=vector(r['original_alpha'],n)
    require(all(0<=v<=C for v in A) and sum(v*t for v,t in zip(A,Y))==0,'exact dual constraints')
    for row,label,slack in zip(X,Y,S):require(slack>=0 and label*(inner(W,row)+B)+slack>=1,'exact primal constraints')
    expected=[sum(O[i]*Y[i]*X[i][j] for i in range(n)) for j in range(d)]
    require(W==expected,'original primal weights')
    # Derive a feature-space lower objective independently, no stored Gram matrix.
    squared=Fraction(0)
    for j in range(d):
        coordinate=sum(A[i]*Y[i]*X[i][j] for i in range(n));squared+=coordinate**2
    upper=sum(v*v for v in W)/2+C*sum(S);lower=sum(A)-squared/2
    require(lower<=upper,'weak duality')
    require(fraction(o['primal'])==upper and fraction(o['dual'])==lower and fraction(o['gap'])==upper-lower,'exact objectives')
    clipped=[min(C,max(Fraction(0),v)) for v in O]
    require(all(v<=z for v,z in zip(A,clipped)),'reduction-only repair')
    require(fraction(r['original_equality'])==sum(v*t for v,t in zip(O,Y)),'original equality')
    require(fraction(r['original_box'])==max([Fraction(0)]+[-v for v in O]+[v-C for v in O]),'original box')
    require(fraction(r['clipped_equality'])==sum(v*t for v,t in zip(clipped,Y)),'clipped equality')
    require(fraction(r['l1_change'])==sum(abs(v-z) for v,z in zip(A,O)),'repair size')
    return {'valid':True,'rows':n,'features':d,'exact_primal':o['primal'],'exact_dual':o['dual'],'exact_gap':o['gap'],'scope':'exact declared represented linear problem only, not source authentication or generalization'}
