"""Independent Fraction active-face KKT oracle; no pair updates or NumPy solves."""
from fractions import Fraction as F
import itertools

def solve(A,b):
    n=len(b);rows=[list(map(F,r))+[F(v)] for r,v in zip(A,b)]
    for col in range(n):
        pivot=next((i for i in range(col,n) if rows[i][col]),None)
        if pivot is None:return None
        rows[col],rows[pivot]=rows[pivot],rows[col];v=rows[col][col];rows[col]=[x/v for x in rows[col]]
        for i in range(n):
            if i!=col:
                v=rows[i][col];rows[i]=[x-v*y for x,y in zip(rows[i],rows[col])]
    return [r[-1] for r in rows]

def optimum(x,y,C):
    C=F(C);n=len(x);y=list(map(F,y));K=[[sum(F(a)*F(b) for a,b in zip(u,v)) for v in x] for u in x];Q=[[y[i]*y[j]*K[i][j] for j in range(n)] for i in range(n)];answers=[]
    for face in itertools.product((0,1,2),repeat=n):
        a=[C if s==2 else F(0) for s in face];free=[i for i,s in enumerate(face) if s==1]
        if free:
            A=[[Q[i][j] for j in free]+[y[i]] for i in free]+[[y[j] for j in free]+[F(0)]]
            b=[1-sum(Q[i][j]*a[j] for j in range(n)) for i in free]+[-sum(y[j]*a[j] for j in range(n))]
            solution=solve(A,b)
            if solution is None:continue
            for i,v in zip(free,solution[:-1]):a[i]=v
            lam=solution[-1]
        else:
            if sum(a[i]*y[i] for i in range(n)):continue
            g=[1-sum(Q[i][j]*a[j] for j in range(n)) for i in range(n)];lower=[];upper=[]
            for i in range(n):
                value=g[i]/y[i]
                (lower if (face[i]==0 and y[i]==1) or (face[i]==2 and y[i]==-1) else upper).append(value)
            lo=max(lower) if lower else None;hi=min(upper) if upper else None
            if lo is not None and hi is not None and lo>hi:continue
            lam=lo if lo is not None else hi if hi is not None else F(0)
        if any(v<0 or v>C for v in a) or sum(a[i]*y[i] for i in range(n)):continue
        g=[1-sum(Q[i][j]*a[j] for j in range(n))-lam*y[i] for i in range(n)]
        if any((a[i]==0 and g[i]>0) or (a[i]==C and g[i]<0) or (0<a[i]<C and g[i]!=0) for i in range(n)):continue
        dual=sum(a)-sum(a[i]*Q[i][j]*a[j] for i in range(n) for j in range(n))/2;answers.append((dual,a,lam))
    if not answers:raise ValueError('oracle found no nonsingular KKT face')
    return max(answers,key=lambda v:v[0])
