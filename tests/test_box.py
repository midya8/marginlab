import copy,itertools,json,subprocess,sys,tempfile,unittest
from fractions import Fraction as F
from pathlib import Path
from marginlab.box import *
from marginlab.box_export import build
from marginlab.experiment import saved

class BoxTests(unittest.TestCase):
    def test_exhaustive_corners(self):
        count=0
        # Independent exhaustive oracle, mixed signs, zeros, offsets, asymmetric bounds.
        for d in range(1,6):
            for seed in range(24):
                w=[F(((seed+3*j)%7)-3,1+j%3) for j in range(d)];b=F(seed-12,7)
                low=[F(-2-j,3) for j in range(d)];high=[F(1+(seed+j)%5,4) for j in range(d)];x=[(a+b)/2 for a,b in zip(low,high)]
                p=problem(dict(weights=list(map(text,w)),bias=text(b)),list(map(text,x)),list(map(text,low)),list(map(text,high)));v=produce(p)
                corners=[b+sum((a*q for a,q in zip(w,point)),F()) for point in itertools.product(*zip(low,high))]
                self.assertEqual(rat(v['minimum']['score']),min(corners));self.assertEqual(rat(v['maximum']['score']),max(corners));self.assertTrue(verify(v)['valid']);count+=1
        self.assertEqual(count,120)
    def test_tie_and_zero(self):
        a=dict(weights=['1/1'],bias='0/1')
        for lo,hi,stable,labels in [('0/1','1/1',True,[1]),('-1/1','0/1',False,[-1,1]),('-1/1','-1/2',True,[-1]),('0/1','0/1',True,[1])]:
            center=text((rat(lo)+rat(hi))/2);p=produce(problem(a,[center],[lo],[hi]));self.assertEqual((p['stable'],p['possible_labels']),(stable,labels))
        p=produce(problem(dict(weights=['0/1'],bias='0/1'),['0/1'],['-1/1'],['2/1']));self.assertTrue(p['stable']);self.assertEqual(p['possible_labels'],[1])
        p['minimum']['point']=['1/1'];self.assertTrue(verify(p)['valid']) # zero coefficient frees the corner
    def test_exact_decimal_binary_distinction(self):
        self.assertNotEqual(decimal('0.1'),binary(.1));self.assertEqual(decimal('-0.1'),F(-1,10));self.assertEqual(binary(-0.),0)
        for s in ['1e-2','NaN',' 1','+1','.1','01','-01','1/2',True]:
            with self.assertRaises(ValueError):decimal(s)
    def test_schema_and_complexity(self):
        good=problem(dict(weights=['1/1'],bias='0/1'),['0/1'],['-1/1'],['1/1'])
        for bad in [dict(good,other=0),dict(good,center=[]),dict(good,center=['2/1']),dict(good,lower=['1/1']),dict(good,upper=['2/2']),dict(good,center=[True]),dict(good,center=['-0/1']),dict(good,center=[1]),dict(good,upper=['1/0']),dict(good,upper=[str(2**513)+'/1']),dict(good,affine=dict(weights=['1/1'],bias='1/0'))]:
            with self.assertRaises((ValueError,TypeError)):produce(bad)
        v=produce(good)
        changes=[('minimum',dict(point=['0/1'],score='-1/1')),('maximum',dict(point=['2/1'],score='2/1')),('center_label',True),('stable',1),('version',True),('possible_labels',[True]),('center_score','1/1')]
        for k,x in changes:
            u=copy.deepcopy(v);u[k]=x
            with self.assertRaises(ValueError):verify(u)
    def test_reference_and_independent_raw_dual(self):
        out=build();r=saved();self.assertEqual(len(out['references']),128)
        for id in out['models']:
            original=json.loads((Path(__file__).parents[1]/('marginlab/artifacts/box-'+id+'.json')).read_text());self.assertEqual(original,[v for v in out['references'] if v['model_id']==id])
        # Direct exact dual score on standardized raw query, independent nested order.
        for ref in out['references']:
            v=ref['proof'];m=next(c['model'] for c in r['candidates'] if c['id']==ref['model_id'])
            for endpoint in ['minimum','maximum']:
                q=list(map(rat,v[endpoint]['point']));z=[(a-binary(mu))/binary(sd) for a,mu,sd in zip(q,r['mean'],r['scale'])];total=binary(m['bias'])
                for i in range(len(m['x'])):
                    kernel=sum((binary(a)*b for a,b in zip(m['x'][i],z)),F())
                    total+=binary(m['alpha'][i])*m['y'][i]*kernel
                self.assertEqual(text(total),v[endpoint]['score'])
            self.assertTrue(verify(v)['valid'])
    def test_nonmutation_source_distinction(self):
        r=saved();old=copy.deepcopy(r);req=dict(center=['0/1']*30,lower=['-1/1']*30,upper=['1/1']*30);p=from_report(r,'linear-1',req);self.assertEqual(r,old)
        changed=copy.deepcopy(p);changed['problem']['affine']['bias']='0/1';new=produce(changed['problem']);self.assertTrue(verify(new)['valid']);self.assertNotEqual(new,p)
        with self.assertRaises(ValueError):from_report(r,'rbf-1',req)
        with self.assertRaises(ValueError):from_report(r,'missing',req)
        m=copy.deepcopy(r['candidates'][0]['model'])
        with self.assertRaises(ValueError):affine(m,r['mean'],[0]*30)
    def test_cli(self):
        with tempfile.TemporaryDirectory() as t:
            root=Path(t);request=root/'request.json';proof=root/'proof.json';request.write_text(json.dumps(dict(center=['0/1']*30,lower=['-1/1']*30,upper=['1/1']*30)))
            a=subprocess.run([sys.executable,'-m','marginlab','box','--input',str(request),'--output',str(proof)],capture_output=True,text=True);self.assertEqual(a.returncode,0,a.stderr)
            b=subprocess.run([sys.executable,'-m','marginlab','verify-box','--input',str(proof)],capture_output=True,text=True);self.assertEqual(b.returncode,0,b.stderr);self.assertTrue(json.loads(b.stdout)['valid'])
            request.write_text('{"center":[],"center":[],"lower":[],"upper":[]}');a=subprocess.run([sys.executable,'-m','marginlab','box','--input',str(request),'--output',str(proof)],capture_output=True,text=True);self.assertEqual(a.returncode,2);self.assertIn('duplicate',a.stderr)
