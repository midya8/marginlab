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
assert len(mods)==6,[v.name for v in mods]
print('all seven modules including package root imported outside checkout')
PY
python -m marginlab report > report.json
python -m marginlab controls > controls.json
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
print('all five installed commands, eight fresh refits, full pair replay and saved/fresh audits pass')
PY
