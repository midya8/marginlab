import unittest,copy,json,hashlib,tempfile
from pathlib import Path
import numpy as np
from marginlab.dataset import load,split,FEATURES,HASHES,DATA
from marginlab.experiment import saved,run,audit,metrics
from marginlab.core import Model

class ExperimentTests(unittest.TestCase):
    def test_original_hashes_counts_and_frozen_stratification(self):
        x,y=load();self.assertEqual(x.shape,(569,30));self.assertEqual(len(FEATURES),30);self.assertEqual(np.count_nonzero(y==1),212)
        ids=split();self.assertEqual([len(a) for a in ids],[341,113,115]);self.assertEqual(sorted(np.concatenate(ids).tolist()),list(range(569)))
        rng=np.random.default_rng(20261108);expected=[[],[],[]]
        for label in (-1,1):
            rows=rng.permutation(np.flatnonzero(y==label));a=int(.6*len(rows));b=int(.8*len(rows))
            for out,part in zip(expected,(rows[:a],rows[a:b],rows[b:])):out.extend(part.tolist())
        for a,b in zip(ids,expected):self.assertEqual(a.tolist(),sorted(b))
    def test_all_eight_fresh_refits_reproduce(self):
        r=run();old=saved();self.assertEqual(r['selected_id'],old['selected_id']);self.assertEqual(r['protocol'],old['protocol'])
        for a,b in zip(r['candidates'],old['candidates']):
            self.assertEqual(a['id'],b['id']);self.assertEqual(a['status'],b['status']);self.assertEqual(a['updates'],b['updates']);np.testing.assert_allclose(a['model']['alpha'],b['model']['alpha'],atol=1e-6,rtol=1e-6)
            for stage in ('train','validation','test'):self.assertEqual(a[stage]['confusion'],b[stage]['confusion'])
        self.assertEqual(len(audit(r)['checks']),8)
    def test_holdout_poison_cannot_change_fits_or_selection(self):
        x,y=load();train,val,test=split();x=x.copy();y=y.copy();x[test]=np.arange(30)+1e6;y[test]=-y[test];a=run(x,y);b=saved()
        self.assertEqual(a['mean'],b['mean']);self.assertEqual(a['scale'],b['scale']);self.assertEqual(a['selected_id'],b['selected_id'])
        for q,r in zip(a['candidates'],b['candidates']):
            self.assertEqual(q['model'],r['model']);self.assertEqual(q['pairs'],r['pairs']);self.assertEqual(q['validation'],r['validation'])
    def test_only_training_rows_scale_and_id_excluded(self):
        x,y=load();train,_,_=split();r=saved();np.testing.assert_allclose(r['mean'],x[train].mean(0));np.testing.assert_allclose(r['scale'],x[train].std(0));self.assertNotIn('ID',FEATURES);self.assertEqual(len(r['mean']),30)
    def test_independent_scores_and_confusions(self):
        x,y=load();r=saved();_,_,test=split();z=(x-np.array(r['mean']))/r['scale']
        for result in r['candidates']:
            m=Model.from_dict(result['model']);s=m.decision(z[test]);wrong=sum((1 if v>=0 else -1)!=int(t) for v,t in zip(s,y[test]));self.assertAlmostEqual(result['test']['accuracy'],1-wrong/len(test))
            self.assertEqual(result['test'],metrics(y[test],s))
    def test_full_pair_replay_and_tamper_rejection(self):
        r=saved();self.assertEqual(len(audit(r)['checks']),8)
        for kind in ('gain','curvature','equality','endpoint','count','status'):
            a=copy.deepcopy(r);q=a['candidates'][0]
            if kind=='gain':q['pairs'][0][5]+=.1
            if kind=='curvature':q['pairs'][0][4]+=.1
            if kind=='equality':q['pairs'][0][2]+=.1
            if kind=='endpoint':q['model']['alpha'][0]+=.1
            if kind=='count':q['updates']+=1
            if kind=='status':q['tolerance']=1e-15
            with self.assertRaises(ValueError,msg=kind):audit(a)
    def test_one_neighbor_uses_all_features_and_original_train(self):
        x,y=load();r=saved();train,val,test=split();z=(x-r['mean'])/r['scale']
        for ids,name in [(val,'validation'),(test,'test')]:
            prediction=[]
            for row in ids:
                distances=[sum((float(a)-float(b))**2 for a,b in zip(z[row],z[k])) for k in train];prediction.append(y[train[int(np.argmin(distances))]])
            self.assertEqual(metrics(y[ids],prediction),r['baselines'][name]['nearest_neighbor'])


    def test_browser_export_numerical_reproduction_only_public_files(self):
        from marginlab.export import build
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);(root/'docs').mkdir();(root/'marginlab/artifacts').mkdir(parents=True);build(root)
            names=sorted(str(p.relative_to(root)) for p in root.rglob('*') if p.is_file());self.assertEqual(names,['docs/data.js','marginlab/artifacts/browser-reference.json'])
            def parse(file):
                s=file.read_text()
                if file.suffix=='.js':s=s.removeprefix('globalThis.MARGIN_DATA = ').rstrip().removesuffix(';')
                return json.loads(s)
            def compare(a,b):
                if isinstance(b,dict):
                    self.assertEqual(set(a),set(b))
                    for k in b:compare(a[k],b[k])
                elif isinstance(b,list):
                    self.assertEqual(len(a),len(b))
                    for q,r in zip(a,b):compare(q,r)
                elif isinstance(b,float):self.assertAlmostEqual(a,b,delta=2e-8*(1+abs(b)))
                else:self.assertEqual(a,b)
            for name in names:compare(parse(root/name),parse(Path(__file__).resolve().parents[1]/name))

if __name__=='__main__':unittest.main()
