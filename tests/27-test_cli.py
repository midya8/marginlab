import unittest,json,tempfile,contextlib,io,sys
from pathlib import Path
from unittest.mock import patch
from marginlab.__main__ import main,read_json
from marginlab.dataset import load

class CLITests(unittest.TestCase):
    def call(self,args):
        out=io.StringIO();err=io.StringIO()
        with patch.object(sys,'argv',['marginlab']+args),contextlib.redirect_stdout(out),contextlib.redirect_stderr(err):code=main()
        return code,out.getvalue(),err.getvalue()
    def test_all_commands_and_explicit_zero_fit(self):
        for command in ('report','controls','audit'):
            code,out,err=self.call([command]);self.assertEqual(code,0);self.assertFalse(err);json.loads(out)
        with tempfile.TemporaryDirectory() as d:
            file=Path(d)/'model.json';code,out,err=self.call(['fit','--budget','0','--output',str(file)]);self.assertEqual(code,0);r=json.loads(file.read_text());self.assertTrue(all(a['status']=='budget_exhausted' and a['updates']==0 for a in r['candidates']))
            sample=Path(d)/'sample.json';sample.write_text(json.dumps({'samples':[load()[0][0].tolist()]}));code,out,err=self.call(['predict','--input',str(sample)]);self.assertEqual(code,0);self.assertEqual(len(json.loads(out)['scores']),1)
    def test_strict_rejections(self):
        row=load()[0][0].tolist()
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/'input.json'
            for text in ('{"samples":[],"samples":[]}','{"samples":[[NaN]]}',json.dumps({'samples':[row],'extra':0}),json.dumps({'samples':[row[:-1]]}),json.dumps({'samples':[[True]+row[1:]]}),json.dumps({'samples':[['3']+row[1:]]})):
                f.write_text(text);code,out,err=self.call(['predict','--input',str(f)]);self.assertEqual(code,2);self.assertEqual(out,'');self.assertTrue(err)
            f.write_text(json.dumps({'samples':[row]}));self.assertEqual(self.call(['predict','--input',str(f),'--model-id','nope'])[0],2)
