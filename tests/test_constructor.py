from .fixtures import *


def test_empty_like(mypytest):
    mypytest('''
import numpy as np
a = np.zeros(10)
b = np.empty_like(a)
c = np.empty_like(a, dtype=bool)
d = np.arange(10, dtype=np.int64)
reveal_type(b)  # Revealed type is 'numpy.ndarray[builtins.float, numpy.OneD]'
reveal_type(c)  # Revealed type is 'numpy.ndarray[builtins.bool, numpy.OneD]'
reveal_type(d)  # Revealed type is 'numpy.ndarray[builtins.int, numpy.OneD]'
''')


def test_array_constructor_0(mypytest):
    mypytest('''
import numpy as np
import itertools
N = 10
l = [(i, i) for i in range(N)]
pairs = np.array(l)
reveal_type(l)      # Revealed type is 'builtins.list[Tuple[builtins.int*, builtins.int*]]'
reveal_type(pairs)  # Revealed type is 'numpy.ndarray[builtins.int, numpy.TwoD]'
reveal_type(np.asarray(np.zeros((3,3))))  # Revealed type is 'numpy.ndarray[builtins.float, numpy.TwoD]'
''')


def test_reshape(mypytest):
    mypytest('''
import numpy as np
a = np.zeros((2,2,2))
reveal_type(np.reshape(a, order='C', newshape=(1,1)))  # Revealed type is 'numpy.ndarray[builtins.float, numpy.TwoD]'
reveal_type(np.reshape(a, 1))  # Revealed type is 'numpy.ndarray[builtins.float, numpy.OneD]'
reveal_type(np.reshape(a, -1))  # Revealed type is 'numpy.ndarray[builtins.float, numpy.OneD]'
reveal_type(a.reshape(order='C', shape=(1,1)))  # Revealed type is 'numpy.ndarray[builtins.float, numpy.TwoD]'
reveal_type(a.reshape(1))  # Revealed type is 'numpy.ndarray[builtins.float, numpy.OneD]'
reveal_type(a.reshape(-1))  # Revealed type is 'numpy.ndarray[builtins.float, numpy.OneD]'
reveal_type(a.reshape((1,1)))  # Revealed type is 'numpy.ndarray[builtins.float, numpy.TwoD]'
''')


def test_vstack(mypytest):
    mypytest('''
import numpy as np
a = np.zeros(10)
b = np.zeros((10, 10))
reveal_type(np.vstack((a, a)))  # Revealed type is 'numpy.ndarray[builtins.float, numpy.TwoD]'
reveal_type(np.vstack((b, b)))  # Revealed type is 'numpy.ndarray[builtins.float, numpy.ThreeD]'
np.vstack((a, b))  # Cannot infer type argument 2 of "vstack"
''')


def test_astype(mypytest):
    mypytest('''
import numpy as np
a = np.zeros(10)
b = a.astype('bool')
reveal_type(b)  # Revealed type is 'numpy.ndarray[builtins.bool, numpy.OneD]'
''')

