#!/usr/bin/env garden-exec
#{
# exec garden with -c         \
#   -e MYPYPATH \
#   -m desres-python/3.6.1-02c7/bin \
#   -s PYTHONPATH=/d/nyc/mcgibbon-0/garden/CentOS7/prefixes/mypy/0.521-01c7/python:$PYTHONPATH \
#   -- python3 "$0" "$@"
#}
import os
import glob
import os.path
from typing import Optional, Callable, List
from myunit.data import parse_test_data


class TestCase:
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def set_up(self):
        with open('test-tmp/%s.py' % self.input.arg, 'w') as f:
            f.write('\n'.join(self.input.data))

        with open('test-tmp/%s.ref' % self.input.arg, 'w') as f:
            f.write('\n'.join(self.output.data))
            if len(self.output.data) > 0:
                f.write('\n')

    def run(self):
        os.system('garden with -m mypy/0.521-01c7/bin mypy --config-file mypy.ini test-tmp/%s.py > test-tmp/%s.out' % 
            (self.input.arg, self.input.arg))
        os.system("sed -i 's|test-tmp/%s.py|main|g' test-tmp/%s.out" % (self.input.arg, self.input.arg))
        print('Case: %s' % self.input.arg)
        os.system('diff test-tmp/%s.ref test-tmp/%s.out' % (self.input.arg, self.input.arg))


def parse_test_cases(fn: str):
    cases = []
    with open(fn) as f:
        p = parse_test_data(f.readlines(), fn)

    i = 0
    input = None
    output = None
    for pp in p:
        if pp.id == 'case':
            input = pp
        if pp.id == 'out':
            output = pp
        if input is not None and output is not None:
            case = TestCase(input, output)
            cases.append(case)
            input = None
            output = None

    return cases


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--cases', nargs='+', default=[])
    args = p.parse_args()
    execute(args, p)

def execute(args, p):
    test_temp_dir = 'test-temp'
    files = glob.glob('tests/*.test')

    cases = []
    for f in files:
        cases += parse_test_cases(f)

    for c in cases:
        if len(args.cases) == 0 or c.input.arg in args.cases:
            c.set_up()
            c.run()


if __name__ == '__main__':
    main()