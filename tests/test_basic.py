from .fixtures import *


def test_zeros_0(mypytest):
    mypytest('''
    import numpy as np
    a = np.zeros(1)
    b = np.zeros((1,2))
    
    reveal_type(a)  # Revealed type is 'numpy.ndarray[builtins.float, numpy.OneD]'
    reveal_type(b)  # Revealed type is 'numpy.ndarray[builtins.float, numpy.TwoD]'
    ''')


def test_zeros_1(mypytest):
    mypytest('''
    import numpy as np
    a = np.zeros([1,2])
    b = np.zeros([1 for _ in range(2+2)])
    reveal_type(a)  # Revealed type is 'numpy.ndarray[builtins.float, numpy.TwoD]'
    reveal_type(b)  # Revealed type is 'numpy.ndarray[builtins.float, Any]'
    ''')

def test_zeros_2(mypytest):
    mypytest('''
import numpy as np
a = np.zeros(1, dtype=int)
b = np.zeros(1, dtype='int')
c = np.zeros(1, dtype='i')
reveal_type(a)  # Revealed type is 'numpy.ndarray[builtins.int, numpy.OneD]'
reveal_type(b)  # Revealed type is 'numpy.ndarray[builtins.int, numpy.OneD]'
reveal_type(c)  # Revealed type is 'numpy.ndarray[builtins.int, numpy.OneD]'
''')


