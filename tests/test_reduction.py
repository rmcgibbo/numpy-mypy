from .fixtures import *


def test_all(mypytest):
    mypytest('''
import numpy as np
reveal_type(np.all(np.zeros(10)))  # Revealed type is 'builtins.bool'
reveal_type(np.all(np.zeros((10,10))))  # Revealed type is 'builtins.bool'
reveal_type(np.all(np.zeros((10,10)), axis=1))  # Revealed type is 'numpy.ndarray[builtins.bool, numpy.OneD]'
reveal_type(np.all(np.zeros((10,10)), axis=(0,1)))  # Revealed type is 'builtins.bool'
reveal_type(np.all(np.zeros((10,10)), axis=(0,1), keepdims=True))  # Revealed type is 'numpy.ndarray[builtins.bool, numpy.TwoD]'

from typing import Any
f = 0  # type: Any
reveal_type(np.all(np.zeros((10,10)), axis=(0,1), keepdims=f))  # Revealed type is 'numpy.ndarray[builtins.bool, Any]'
reveal_type(np.all(np.zeros((10,10)), axis=(0,1), keepdims=False))  # Revealed type is 'builtins.bool'
''')


def test_choose(mypytest):
    mypytest('''
import numpy as np
a = np.zeros(10, dtype='int')
b = np.choose(a, (np.ones(10), np.zeros(10)))
reveal_type(b)  # Revealed type is 'numpy.ndarray[Any, numpy.OneD*]'
''')


def test_cumsum(mypytest):
    mypytest('''
import numpy as np
reveal_type(np.cumsum(np.zeros((10,10)),axis=1))  # Revealed type is 'numpy.ndarray[builtins.float*, numpy.TwoD*]'
reveal_type(np.cumsum(np.zeros((10,10))))  # Revealed type is 'numpy.ndarray[builtins.float*, numpy.OneD]'
reveal_type(np.cumsum(np.zeros((10,10)),axis=None))  # Revealed type is 'numpy.ndarray[builtins.float*, numpy.OneD]'
''')


def test_diagonal(mypytest):
    mypytest('''
import numpy as np
reveal_type(np.diagonal(np.zeros((10,10)))) # Revealed type is 'numpy.ndarray[builtins.float, numpy.OneD]'
reveal_type(np.diagonal(np.zeros((10,10,10)))) # Revealed type is 'numpy.ndarray[builtins.float, numpy.TwoD]'
''')


def test_diag(mypytest):
    mypytest('''
import numpy as np
reveal_type(np.diag(np.zeros((10,10))))  # Revealed type is 'numpy.ndarray[builtins.float, numpy.OneD]'
reveal_type(np.diag(np.zeros(10)))  # Revealed type is 'numpy.ndarray[builtins.float, numpy.TwoD]'
''')
