import os
import pytest
import tempfile
import subprocess
import shutil
from distutils.spawn import find_executable

BASE_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))


@pytest.fixture
def mypytest():
    return _mypytest_inner


def _mypytest_inner(s: str):
    input_lines = [e.lstrip() for e in s.splitlines()]
    expected_output = []
    for i, line in enumerate(input_lines):
        if '# Revealed type is' in line:
            expected_output.append(line[line.find('# Revealed type is'):][2:])

    td = tempfile.mkdtemp()
    with open(os.path.join(td, 'input.py'), 'w') as f:
        f.write('\n'.join(input_lines))
    with open(os.path.join(td, 'mypy.ini'), 'w') as f:
        f.write('''
[mypy]

mypy_path = {0}
incremental = True
disallow_untyped_defs = True
ignore_missing_imports = True
show_traceback = True
plugins = {0}/numpy_plugin_entry.py'''.format(BASE_DIR))


    curdir = os.path.abspath(os.curdir)
    try:
        os.chdir(td)
        cmd = ['mypy', '--config-file', 'mypy.ini', 'input.py']
        p = subprocess.run(' '.join(cmd), stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True, shell=True)
        received_output = []
        for l in p.stdout.splitlines():
            received_output.append(l.split('error: ')[1])

    finally:
        os.chdir(curdir)
        shutil.rmtree(td)

    print('\n'.join(received_output))
    assert len(expected_output) == len(received_output), (len(expected_output), len(received_output))
    for e, r in zip(expected_output, received_output):
        assert e == r, ('"%s" != "%s"' % (e, r))


