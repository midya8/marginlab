"""Reproduce small exact affine declarations and full source-bound reference responses."""
import json
from fractions import Fraction as F
from pathlib import Path
from .box import affine,binary,text,produce,problem
from .experiment import saved
from .dataset import load,split

def build():
    r=saved();raw,_=load();ids=split()[2][:8].tolist();models={};refs=[]
    for c in r['candidates']:
        if c['model']['kind']!='linear':continue
        a=affine(c['model'],r['mean'],r['scale']);models[c['id']]=a
        for i in ids:
            x=list(map(binary,raw[i].tolist()))
            for eps in [F(0),F(1,100),F(1,10),F(1)]:
                h=[binary(v)*eps for v in r['scale']]
                p=problem(a,list(map(text,x)),list(map(text,[v-z for v,z in zip(x,h)])),list(map(text,[v+z for v,z in zip(x,h)])))
                refs.append(dict(model_id=c['id'],row_index=i,epsilon=text(eps),proof=produce(p)))
    return dict(version=1,models=models,row_indices=ids,references=refs)

def main():
    root=Path(__file__).resolve().parents[1];out=build()
    for id in out['models']:
        chunk=[v for v in out['references'] if v['model_id']==id]
        (root/('marginlab/artifacts/box-'+id+'.json')).write_text(json.dumps(chunk,separators=(',',':'),allow_nan=False)+'\n')
    small={k:v for k,v in out.items() if k!='references'}
    (root/'docs/box-data.js').write_text('globalThis.MARGIN_BOX_DATA = '+json.dumps(small,separators=(',',':'),allow_nan=False)+';\n')
    print('128 complete exact responses, four original-source affine declarations')
if __name__=='__main__':main()
