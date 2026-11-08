import unittest,math,random,copy
from fractions import Fraction
import numpy as np
from marginlab.core import fit,Model,kernel,audit_gram
from marginlab.controls import controls
from oracle import optimum

class CoreTests(unittest.TestCase):
    def test_sixty_exact_active_face_oracles(self):
        rng=random.Random(180);count=0
        for _ in range(60):
            x=[[rng.randint(-3,3),rng.randint(-3,3)] for i in range(4)];y=[-1,-1,1,1];C=rng.choice([Fraction(1,5),Fraction(1,2),Fraction(1),Fraction(3)])
            truth,a,b=optimum(x,y,C);actual=fit(x,y,float(C),tolerance=1e-9,max_updates=20000)
            self.assertAlmostEqual(actual['audit']['dual'],float(truth),delta=2e-7)
            self.assertLessEqual(actual['audit']['gap'],2e-6);count+=1
        self.assertEqual(count,60)
    def test_kernel_independent_scalar_and_psd(self):
        x=np.array([[1.,2.],[-3,4],[0,0],[1,2]])
        for kind,gamma in [('linear',None),('rbf',.3)]:
            K=kernel(x,x,kind,gamma)
            for i in range(4):
                for j in range(4):self.assertAlmostEqual(K[i,j],sum(a*b for a,b in zip(x[i],x[j])) if kind=='linear' else math.exp(-.3*sum((a-b)**2 for a,b in zip(x[i],x[j]))),places=14)
            self.assertGreaterEqual(np.linalg.eigvalsh(K)[0],-1e-12)
    def test_analytic_two_point_and_zero_curvature(self):
        for C in (.1,1,5):
            a=fit([[-1.],[1.]],[-1,1],C);self.assertTrue(np.allclose(a['model']['alpha'],[min(C,.5)]*2));self.assertAlmostEqual(a['audit']['gap'],0)
        a=controls()['zero_curvature'];self.assertEqual(a['updates'],1);self.assertEqual(a['pairs'][0][4],0);self.assertAlmostEqual(a['audit']['dual'],1.4);self.assertEqual(a['audit']['gap'],0)
    def test_accuracy_is_not_optimization(self):
        c=controls()['accuracy_not_optimization'];self.assertEqual(c['short_accuracy'],1);self.assertEqual(c['short']['status'],'budget_exhausted');self.assertGreater(c['short']['audit']['gap'],.1);self.assertLess(c['full']['audit']['gap'],1e-8)
    def test_duplicate_alpha_is_not_unique_data_importance(self):
        c=controls()['duplicate_support'];self.assertEqual(c['audit_a']['dual'],c['audit_b']['dual']);self.assertEqual(c['audit_a']['gap'],0);self.assertEqual(c['audit_b']['gap'],0)
        x=np.array(c['x']);y=np.array(c['y']);self.assertTrue(np.array_equal(np.array(c['alpha_a'])*y@x,np.array(c['alpha_b'])*y@x))
    def test_label_and_input_symmetries(self):
        x=np.array([[-2,1],[-1,0],[1,0],[2,1]],dtype=float);y=np.array([-1,-1,1,1]);a=fit(x,y);m=Model.from_dict(a['model']);b=Model.from_dict(fit(x,-y)['model']);np.testing.assert_allclose(m.decision(x),-b.decision(x),atol=1e-7)
        R=np.array([[0,-1],[1,0]]);q=Model.from_dict(fit(x@R,y)['model']);np.testing.assert_allclose(q.decision(x@R),m.decision(x),atol=1e-7)
        q=Model.from_dict(fit(x+np.array([3,7]),y)['model']);np.testing.assert_allclose(q.decision(x+[3,7]),m.decision(x),atol=1e-7)
    def test_budget_zero_and_strict_failures(self):
        a=fit([[-1],[1]],[-1,1],max_updates=0);self.assertEqual(a['updates'],0);self.assertEqual(a['status'],'budget_exhausted');self.assertEqual(a['audit']['dual'],0)
        for kw in ({'C':0},{'C':math.nan},{'max_updates':True},{'max_updates':-1},{'tolerance':0},{'kind':'invalid'},{'kind':'rbf','gamma':0}):
            with self.assertRaises(ValueError):fit([[-1],[1]],[-1,1],**kw)
        for x,y in (([[0],[1]],[1,1]),([[0],[math.inf]],[-1,1]),([[1]],[-1,1])):
            with self.assertRaises(ValueError):fit(x,y)
    def test_model_roundtrip_and_decision_not_probability(self):
        a=fit([[-1],[1]],[-1,1]);m=Model.from_dict(a['model']);self.assertEqual(m.to_dict(),a['model']);self.assertGreater(m.decision([[100]])[0],1)

if __name__=='__main__':unittest.main()
