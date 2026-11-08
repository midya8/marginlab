import unittest,copy,random,json,subprocess,sys,tempfile
from pathlib import Path
from fractions import Fraction as F
from marginlab.certificate import produce,verify,rational
from marginlab.verify_certificate import check
from marginlab.experiment import saved
from marginlab.core import fit
from oracle import optimum

class CertificateTests(unittest.TestCase):
    def test_four_real_linear_proofs_independently_verified(self):
        for candidate in saved()['candidates'][:4]:
            proof=produce(candidate['model']);a=verify(proof);b=check(proof)
            self.assertEqual(a['exact_gap'],b['exact_gap']);self.assertGreaterEqual(rational(a['exact_gap']),0)
            self.assertEqual(b['rows'],341);self.assertEqual(b['features'],30)
            self.assertLess(float(rational(a['exact_gap'])),.001)
            self.assertNotEqual(rational(proof['repair']['original_equality']),0)
    def test_sixty_independent_exact_optima_enclosed(self):
        rng=random.Random(981);count=0
        for _ in range(60):
            x=[[rng.randint(-3,3),rng.randint(-3,3)] for _ in range(4)];y=[-1,-1,1,1];C=rng.choice([.125,.5,1.,3.])
            # These C values and integer features are exactly binary64.
            truth,_,_=optimum(x,y,F.from_float(C));model=fit(x,y,C,tolerance=1e-9)['model'];p=produce(model);check(p)
            self.assertLessEqual(rational(p['objectives']['dual']),truth);self.assertGreaterEqual(rational(p['objectives']['primal']),truth);count+=1
        self.assertEqual(count,60)
    def test_repair_both_signs_and_box_then_equality(self):
        base=fit([[-1.],[1.]],[-1,1])['model']
        for alpha in ([.5,.5000000000000001],[.5000000000000001,.5],[-1e-12,1.000000000001],[1.000000000001,-1e-12]):
            m=copy.deepcopy(base);m['alpha']=list(alpha);p=produce(m);check(p);a=list(map(rational,p['dual']['alpha']));self.assertEqual(a[0],a[1]);self.assertTrue(all(0<=v<=1 for v in a))
    def test_budget_zero_and_zero_curvature(self):
        for x,y,C,budget in [([[-1.],[1.]],[-1,1],1.,0),([[2.],[2.]],[-1,1],.5,100)]:
            p=produce(fit(x,y,C,max_updates=budget)['model']);check(p)
            if budget==0:self.assertGreater(rational(p['objectives']['gap']),0)
            else:self.assertEqual(rational(p['objectives']['gap']),0)
    def test_tampering_rejected_by_both_readers(self):
        p=produce(fit([[-1.],[1.]],[-1,1])['model'])
        for mutate in [lambda q:q['dual']['alpha'].__setitem__(0,'3/4'),lambda q:q['primal']['w'].__setitem__(0,'0/1'),lambda q:q['primal']['slacks'].__setitem__(0,'-1/1'),lambda q:q['objectives'].__setitem__('gap','1/1'),lambda q:q['repair'].__setitem__('l1_change','1/1'),lambda q:q['repair']['original_alpha'].__setitem__(0,'3/4'),lambda q:q['problem']['x'][0].__setitem__(0,-2.),lambda q:q['problem'].__setitem__('C',.25),lambda q:q.__setitem__('version',True),lambda q:q['dual']['alpha'].__setitem__(0,'02/4')]:
            q=copy.deepcopy(p);mutate(q)
            for reader in (verify,check):
                with self.assertRaises((ValueError,TypeError,KeyError)):reader(q)
    def test_explicit_unsupported_kernel_and_strict_values(self):
        with self.assertRaises(ValueError):produce(fit([[-1.],[1.]],[-1,1],kind='rbf',gamma=1.)['model'])
        for value in ('1/0','2/4','+1/2','1/-2','1/'+str(1<<5000),False):
            with self.assertRaises(ValueError):rational(value)
        for value in (True,float('nan'),float('inf'),'1'):
            m=fit([[-1.],[1.]],[-1,1])['model'];m['x'][0][0]=value
            with self.assertRaises(ValueError):produce(m)

    def test_cli_roundtrip_tamper_unsupported_and_duplicate_keys(self):
        def call(*args):return subprocess.run([sys.executable,'-m','marginlab',*args],text=True,capture_output=True)
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/'proof.json';a=call('certificate','--output',str(file));self.assertEqual(a.returncode,0,a.stderr);self.assertTrue(json.loads(a.stdout)['valid'])
            b=call('verify-certificate','--input',str(file));self.assertEqual(b.returncode,0,b.stderr);self.assertTrue(json.loads(b.stdout)['valid'])
            p=json.loads(file.read_text());p['objectives']['gap']='0/1';file.write_text(json.dumps(p));self.assertEqual(call('verify-certificate','--input',str(file)).returncode,2)
            a=call('certificate','--model-id','rbf-1','--output',str(file));self.assertEqual(a.returncode,2);self.assertIn('linear kernels only',a.stderr)
            file.write_text('{"version":1,"version":2}');self.assertEqual(call('verify-certificate','--input',str(file)).returncode,2)

    def test_validity_is_not_problem_source_authentication(self):
        proof=produce(saved()['candidates'][2]['model'])
        self.assertEqual(proof['repair']['original_alpha'][0],'0/1')
        # Changing an inactive row can leave every exact feasibility/objective check true.
        # The proof still bounds that changed declared problem; it does not authenticate UCI.
        proof['problem']['x'][0][0]=42.
        self.assertTrue(verify(proof)['valid']);self.assertTrue(check(proof)['valid'])

    def test_proof_export_reproduces_exact_files_without_refits(self):
        import importlib.util
        root=Path(__file__).resolve().parents[1]
        spec=importlib.util.spec_from_file_location('certificate_export',root/'examples/certificate_export.py');module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as d:
            target=Path(d);(target/'docs').mkdir();names=module.build(target);self.assertEqual(len(names),4)
            for name in names:self.assertEqual((target/'docs'/name).read_bytes(),(root/'docs'/name).read_bytes())
