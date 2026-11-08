"""Export full self-contained rational proofs. Never refit or edit old artifacts."""
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from marginlab.experiment import saved
from marginlab.certificate import produce
ROOT=Path(__file__).resolve().parents[1]
def build(root=ROOT):
    report=saved();out=[]
    for c in report['candidates']:
        if c['model']['kind']!='linear':continue
        p=produce(c['model']);name='certificate-data-'+c['id']+'.js';data='globalThis.MARGIN_CERTIFICATES ??= {};\nglobalThis.MARGIN_CERTIFICATES['+json.dumps(c['id'])+'] = '+json.dumps(p,separators=(',',':'),allow_nan=False)+';\n';(root/'docs'/name).write_text(data);out.append(name)
    return out
if __name__=='__main__':print(build())
