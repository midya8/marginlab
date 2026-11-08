"""Reproducible held-out score figure; no fit or model selection."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def svg(report):
    rows=[(a['id']+(' *' if a['id']==report['selected_id'] else ''),a['test']['accuracy']) for a in report['candidates']]+[('1NN, all 30 features',report['baselines']['test']['nearest_neighbor']['accuracy']),('Training-majority',report['baselines']['test']['majority']['accuracy'])]
    s=['<svg xmlns="http://www.w3.org/2000/svg" width="1000" height="850" viewBox="0 0 1000 850"><rect width="1000" height="850" fill="#fbfcf8"/>','<g font-family="sans-serif" fill="#193b36"><text x="40" y="55" font-size="30">Same 115 test rows. One validation choice.</text><text x="40" y="90" font-size="17">Test accuracy, common 0-100% scale. * Validation-selected model, not best test.</text>']
    for i,(name,value) in enumerate(rows):
        y=145+i*55;s.append(f'<text x="40" y="{y+20}" font-size="17">{name}</text><rect data-score="{value!r}" x="300" y="{y}" width="{value*540:.12f}" height="27" fill="'+('#925593' if name.endswith('*') else '#286b5b')+f'"/><text x="855" y="{y+20}" font-size="17">{value*100:.2f}%</text>')
    s+=['<text x="300" y="725" font-size="16">0%</text><text x="790" y="725" font-size="16">100%</text><text x="40" y="775" font-size="17">Small historical UCI split; not clinical validation, calibrated certainty or a general ranking.</text><text x="40" y="808" font-size="16">Source: original UCI Wisconsin Diagnostic data, CC BY 4.0. Original implementation.</text></g></svg>'];return '\n'.join(s)+'\n'
if __name__=='__main__':(ROOT/'assets/test-accuracy.svg').write_text(svg(json.loads((ROOT/'marginlab/artifacts/report.json').read_text())))
