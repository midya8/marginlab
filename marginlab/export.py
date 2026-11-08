"""Browser references and data, no refits."""
import json
from pathlib import Path
import numpy as np
from .experiment import saved
from .dataset import load,split
from .core import Model
from .controls import controls

def build(root):
    root=Path(root);r=saved();raw,y=load();train,val,test=split();z=(raw-r['mean'])/r['scale'];models=[];refs=[]
    for a in r['candidates']:
        m=Model.from_dict(a['model']);models.append(dict(id=a['id'],kind=m.kind,C=m.C,gamma=m.gamma,alpha=m.alpha.tolist(),bias=m.bias,status=a['status'],updates=a['updates'],audit=a['audit'],history=a['history'],train=a['train'],validation=a['validation'],test=a['test']))
        refs.append(dict(id=a['id'],scores=m.decision(z).tolist()))
    data=dict(features=r['features'],mean=r['mean'],scale=r['scale'],raw=raw.tolist(),labels=y.tolist(),train_ids=train.tolist(),split_ids=r['protocol']['split_ids'],models=models,selected_id=r['selected_id'],controls=controls(),baselines=r['baselines'],protocol=r['protocol'])
    (root/'docs/data.js').write_text('globalThis.MARGIN_DATA = '+json.dumps(data,separators=(',',':'),allow_nan=False)+';\n')
    (root/'marginlab/artifacts/browser-reference.json').write_text(json.dumps(refs,separators=(',',':'),allow_nan=False)+'\n')
    return data
if __name__=='__main__':build(Path(__file__).resolve().parents[1])
