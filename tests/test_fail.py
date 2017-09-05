from .fixtures import *


def test_fail_0(mypytest):
    mypytest('''
    import numpy as np
    np.zeros((3,3))[1,2,3]
''')