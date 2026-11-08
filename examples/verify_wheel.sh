#!/usr/bin/env bash
set -euo pipefail
installed=$1
source_root=$2
work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cd "$work"
export PYTHONPATH="$installed"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
python - <<'PY'
import importlib,pkgutil,marginlab
mods=list(pkgutil.walk_packages(marginlab.__path__,marginlab.__name__+'.'))
for m in mods:importlib.import_module(m.name)
assert len(mods)==10,[v.name for v in mods]
print('all eleven modules including package root imported outside checkout')
PY
python -m marginlab report > report.json
python -m marginlab controls > controls.json
python -m marginlab certificate --output proof.json > certificate.json
python -m marginlab verify-certificate --input proof.json > verified.json
cp "$source_root/examples/box.json" box-input.json
python -m marginlab box --input box-input.json --output box-proof.json > box.json
python -m marginlab verify-box --input box-proof.json > box-verified.json
python - <<'PY'
from marginlab.box_export import build
from marginlab.box import verify
refs=build()["references"];assert len(refs)==128
assert all(verify(v["proof"])["valid"] for v in refs)
PY
python -m marginlab audit > audit.json
python -m marginlab fit --output fresh.json > fit-summary.json
python -m marginlab audit --report fresh.json > fresh-audit.json
python - <<'PY'
import json
from marginlab.dataset import load
json.dump({'samples':load()[0][:2].tolist()},open('samples.json','w'))
PY
python -m marginlab predict --input samples.json > prediction.json
python - <<'PY'
import json
from marginlab.experiment import saved
old=saved();fresh=json.load(open('fresh.json'));assert old['selected_id']==fresh['selected_id']
for a,b in zip(old['candidates'],fresh['candidates']):
 assert a['status']==b['status']
 assert a['test']['confusion']==b['test']['confusion']
assert len(json.load(open('prediction.json'))['scores'])==2
assert json.load(open('verified.json'))['valid'] is True
assert json.load(open('box-verified.json'))['valid'] is True
print('all nine main commands plus box exporter,128full box references, eight fresh refits, full pair replay and saved/fresh audits pass')
PY
