"""Original UCI data, IDs excluded; frozen class-stratified split."""
import csv,hashlib,json
from pathlib import Path
import numpy as np
DATA=Path(__file__).parent/'data'
HASHES={'wdbc.data':'d606af411f3e5be8a317a5a8b652b425aaf0ff38ca683d5327ffff94c3695f4a','wdbc.names':'840e04e3f20f8a5b326892f3b9cbc01c4cd6f7e6c597630b701ef6c0ac79f5ef'}
FEATURES=[stat+' '+name for stat in ('mean','standard error','worst') for name in ('radius','texture','perimeter','area','smoothness','compactness','concavity','concave points','symmetry','fractal dimension')]
def load():
    for name,h in HASHES.items():
        source=json.loads((DATA/'wdbc.names.json').read_text()).encode() if name=='wdbc.names' else (DATA/name).read_bytes()
        if hashlib.sha256(source).hexdigest()!=h:raise ValueError('source hash differs: '+name)
    rows=list(csv.reader((DATA/'wdbc.data').open()));x=np.array([[float(v) for v in row[2:]] for row in rows]);y=np.array([1 if row[1]=='M' else -1 for row in rows])
    if x.shape!=(569,30) or set(y)!={-1,1} or not np.isfinite(x).all():raise ValueError('unexpected original data')
    return x,y

def split():
    import json
    file=DATA/'split.json'
    if hashlib.sha256(file.read_bytes()).hexdigest()!='4e715d56dd130c978c351084534272df9df8fb1884d44db90c5b0597e909c40d':raise ValueError('frozen split hash differs')
    values=json.loads(file.read_text())['split_ids']
    return [np.array(values[k],dtype=int) for k in ('train','validation','test')]
