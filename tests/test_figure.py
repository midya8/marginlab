import unittest,re,importlib.util
from pathlib import Path
from marginlab.experiment import saved
ROOT=Path(__file__).resolve().parents[1]
class FigureTests(unittest.TestCase):
    def test_reproducible_all_eight_and_controls_on_zero_scale(self):
        spec=importlib.util.spec_from_file_location('figure',ROOT/'examples/figure.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);s=(ROOT/'assets/test-accuracy.svg').read_text();self.assertEqual(m.svg(saved()),s)
        bars=re.findall(r'data-score="([^"]+)" x="300" y="[^"]+" width="([^"]+)"',s);self.assertEqual(len(bars),10)
        for score,width in bars:self.assertAlmostEqual(float(width),540*float(score),places=10)

        r=saved();expected=[a['test']['accuracy'] for a in r['candidates']]+[r['baselines']['test'][name]['accuracy'] for name in ['nearest_neighbor','majority']]
        self.assertEqual([float(score) for score,width in bars],expected)
