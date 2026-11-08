'use strict';
(function(root){
function check(p){
 const need=(b,s)=>{if(!b)throw Error(s);};
 const gcd=(a,b)=>{a=a<0n?-a:a;while(b){const r=a%b;a=b;b=r;}return a;};
 const norm=(n,d=1n)=>{need(d!==0n,'zero denominator');if(d<0n){n=-n;d=-d;}const g=gcd(n,d);return [n/g,d/g];};
 const add=(a,b)=>norm(a[0]*b[1]+b[0]*a[1],a[1]*b[1]);
 const neg=a=>[-a[0],a[1]],sub=(a,b)=>add(a,neg(b)),mul=(a,b)=>norm(a[0]*b[0],a[1]*b[1]);
 const cmp=(a,b)=>{const x=a[0]*b[1]-b[0]*a[1];return x<0n?-1:x>0n?1:0;},eq=(a,b)=>cmp(a,b)===0;
 const Z=[0n,1n],ONE=[1n,1n],HALF=[1n,2n],abs=a=>a[0]<0n?neg(a):a,min=(a,b)=>cmp(a,b)<0?a:b,max=(a,b)=>cmp(a,b)>0?a:b;
 const sum=a=>a.reduce(add,Z),dot=(a,b)=>sum(a.map((v,i)=>mul(v,b[i]))),txt=a=>a[0]+'/'+a[1];
 const rational=s=>{need(typeof s==='string'&&s.length<=2500&&/^-?\d+\/[1-9]\d*$/.test(s),'rational syntax');const [n,d]=s.split('/').map(BigInt),a=norm(n,d);need(txt(a)===s&&n.toString(2).replace('-','').length<=4096&&d.toString(2).length<=4096,'canonical bounded rational');return a;};
 const binary=v=>{need(typeof v==='number'&&Number.isFinite(v),'finite numeric problem');const b=new ArrayBuffer(8),view=new DataView(b);view.setFloat64(0,v,false);const hi=view.getUint32(0,false),lo=view.getUint32(4,false),sign=hi>>>31?-1n:1n,exponent=(hi>>>20)&2047;let mantissa=(BigInt(hi&0xfffff)<<32n)|BigInt(lo);if(exponent)mantissa|=1n<<52n;const power=(exponent?exponent-1023:1-1023)-52;return power>=0?norm(sign*mantissa*(1n<<BigInt(power))):norm(sign*mantissa,1n<<BigInt(-power));};
 const keys=(o,list)=>need(o&&typeof o==='object'&&!Array.isArray(o)&&Object.keys(o).sort().join('|')===list.slice().sort().join('|'),'schema');
 const vector=(a,n)=>{need(Array.isArray(a)&&a.length===n,'vector dimension');return a.map(rational);};
 keys(p,['version','problem','primal','dual','objectives','repair']);need(p.version===1,'version');keys(p.problem,['x','y','C']);const raw=p.problem.x,y=p.problem.y;need(Array.isArray(raw)&&raw.length>0&&raw.length<=1000,'rows');const n=raw.length;need(Array.isArray(raw[0])&&raw[0].length>0&&raw[0].length<=100,'features');const d=raw[0].length;need(raw.every(row=>Array.isArray(row)&&row.length===d),'feature dimensions');need(Array.isArray(y)&&y.length===n&&y.every(v=>v===-1||v===1),'labels');const X=raw.map(row=>row.map(binary)),Y=y.map(binary),C=binary(p.problem.C);need(cmp(C,Z)>0,'positive C');
 keys(p.primal,['w','bias','slacks']);keys(p.dual,['alpha']);keys(p.objectives,['primal','dual','gap']);keys(p.repair,['original_alpha','original_equality','original_box','clipped_equality','l1_change']);
 const W=vector(p.primal.w,d),B=rational(p.primal.bias),S=vector(p.primal.slacks,n),A=vector(p.dual.alpha,n),O=vector(p.repair.original_alpha,n);
 need(A.every(v=>cmp(v,Z)>=0&&cmp(v,C)<=0)&&eq(dot(A,Y),Z),'exact dual constraints');
 need(S.every((s,i)=>cmp(s,Z)>=0&&cmp(add(mul(Y[i],add(dot(W,X[i]),B)),s),ONE)>=0),'exact primal constraints');
 const originalW=Array.from({length:d},(_,j)=>sum(O.map((a,i)=>mul(mul(a,Y[i]),X[i][j]))));need(W.every((w,j)=>eq(w,originalW[j])),'original primal weights');
 const dualW=Array.from({length:d},(_,j)=>sum(A.map((a,i)=>mul(mul(a,Y[i]),X[i][j]))));
 const P=add(mul(dot(W,W),HALF),mul(C,sum(S))),D=sub(sum(A),mul(dot(dualW,dualW),HALF)),gap=sub(P,D);need(cmp(P,D)>=0,'weak duality');
 need(eq(rational(p.objectives.primal),P)&&eq(rational(p.objectives.dual),D)&&eq(rational(p.objectives.gap),gap),'exact objectives');
 const clipped=O.map(v=>min(C,max(Z,v)));need(A.every((a,i)=>cmp(a,clipped[i])<=0),'reduction-only repair');
 const originalBox=O.reduce((a,v)=>max(a,max(neg(v),sub(v,C))),Z),movement=sum(O.map((a,i)=>abs(sub(a,A[i]))));
 need(eq(rational(p.repair.original_equality),dot(O,Y))&&eq(rational(p.repair.original_box),originalBox)&&eq(rational(p.repair.clipped_equality),dot(clipped,Y))&&eq(rational(p.repair.l1_change),movement),'repair accounting');
 return {valid:true,rows:n,features:d,exact_primal:txt(P),exact_dual:txt(D),exact_gap:txt(gap),repaired_rows:O.filter((v,i)=>!eq(v,A[i])).length,scope:'exact declared represented linear problem only, not source authentication, RBF or generalization'};
}
const api={check};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.MarginCertificate=api;
})(globalThis);
