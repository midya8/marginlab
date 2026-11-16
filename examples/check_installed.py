"""Run the full suite and CLI replay from a wheel outside the checkout."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import venv

ROOT = Path(__file__).resolve().parents[1]
ENV = os.environ.copy()
ENV.pop('PYTHONPATH', None)
ENV.update(OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1')
MODULES = ['marginlab', 'marginlab.__main__', 'marginlab.box', 'marginlab.box_export',
           'marginlab.certificate', 'marginlab.controls', 'marginlab.core',
           'marginlab.dataset', 'marginlab.experiment', 'marginlab.export',
           'marginlab.verify_certificate']
COMMANDS = [
    ['report'], ['controls'], ['audit'],
    ['certificate', '--output', 'proof.json'],
    ['verify-certificate', '--input', 'proof.json'],
    ['box', '--input', 'examples/box.json', '--output', 'box-proof.json'],
    ['verify-box', '--input', 'box-proof.json'],
    ['fit', '--output', 'fresh.json'],
    ['audit', '--report', 'fresh.json'],
    ['predict', '--input', 'samples.json'],
    ['predict', '--input', 'samples.json', '--report', 'fresh.json'],
]


def call(args, cwd, env=ENV):
    return subprocess.check_output(args, cwd=cwd, env=env, text=True)


with tempfile.TemporaryDirectory() as directory:
    work = Path(directory)
    wheels = work / 'wheels'
    wheels.mkdir()
    call([sys.executable, '-m', 'pip', 'wheel', '--no-deps', '-w', str(wheels),
          str(ROOT)], work)
    built = list(wheels.glob('*.whl'))
    assert len(built) == 1, built
    venv.create(work / 'env', with_pip=True)
    py = str(work / 'env/bin/python')
    call([py, '-m', 'pip', 'install', str(built[0])], work)
    outside = work / 'outside'
    outside.mkdir()
    for name in ['tests', 'examples', 'assets', 'docs']:
        shutil.copytree(ROOT / name, outside / name)
    # The tests read these archive paths. No library Python files are copied.
    shutil.copytree(ROOT / 'marginlab/artifacts', outside / 'marginlab/artifacts')
    suite = '''
import importlib,importlib.util,json,pathlib,sys,unittest
outside=pathlib.Path.cwd().resolve()
sys.path.insert(0,str(outside))
names=json.loads(sys.argv[1])
source=pathlib.Path(sys.argv[2]).resolve()
prefix=pathlib.Path(sys.prefix).resolve()
for name in names:importlib.import_module(name)
def check():
    for name in names:
        origin=pathlib.Path(importlib.util.find_spec(name).origin).resolve()
        assert origin.is_relative_to(prefix),(name,str(origin))
        assert not origin.is_relative_to(source),(name,str(origin))
        assert 'site-packages' in origin.parts,(name,str(origin))
check()
tests=unittest.defaultTestLoader.discover(str(outside/'tests'))
result=unittest.TextTestRunner(verbosity=2).run(tests)
assert result.testsRun>0 and result.wasSuccessful()
check()
print(json.dumps(dict(tests=result.testsRun,installed_origins=len(names),
                     checked_before_and_after=True)))
'''
    print(call([py, '-I', '-c', suite, json.dumps(MODULES), str(ROOT)], outside))
    reference = work / 'reference'
    reference.mkdir()
    shutil.copytree(ROOT / 'examples', reference / 'examples')
    source_env = ENV.copy()
    source_env['PYTHONPATH'] = str(ROOT)
    source = [sys.executable, '-m', 'marginlab']
    sample_code = "import json;from marginlab.dataset import load;print(json.dumps({'samples':load()[0][:2].tolist()}))"
    samples = call([sys.executable, '-c', sample_code], reference, source_env)
    for folder in [reference, outside]:
        (folder / 'samples.json').write_text(samples)
    expected = [json.loads(call(source + args, reference, source_env)) for args in COMMANDS]
    # Fresh files, not merely the summaries printed by their commands.
    files = ['fresh.json', 'proof.json', 'box-proof.json']
    saved_files = {name: json.loads((reference / name).read_text()) for name in files}
    entries = [[py, '-I', '-m', 'marginlab'], [str(work / 'env/bin/marginlab')]]
    for entry in entries:
        for args, report in zip(COMMANDS, expected):
            assert json.loads(call(entry + args, outside)) == report, (entry, args)
        for name, report in saved_files.items():
            assert json.loads((outside / name).read_text()) == report, (entry, name)
    print(json.dumps(dict(complete_reports=len(COMMANDS), command_kinds=9,
                          entrypoints=2, complete_output_files=len(files),
                          fresh_models=8, reproduced_figures=1)))
