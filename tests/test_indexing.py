from .fixtures import *


def test_indexing_0(mypytest):
    mypytest('''
import numpy as np
a = np.zeros(10, dtype='int')
b = np.zeros((10, 10))

reveal_type(b[0])         # Revealed type is 'numpy.ndarray[builtins.float*, numpy.OneD]'
reveal_type(b[0,0])       # Revealed type is 'builtins.float*'
reveal_type(b[a])         # Revealed type is 'numpy.ndarray[builtins.float*, numpy.TwoD]'
reveal_type(b[:, :])      # Revealed type is 'numpy.ndarray[builtins.float*, numpy.TwoD]'
reveal_type(b[...])       # Revealed type is 'numpy.ndarray[builtins.float*, numpy.TwoD]'
reveal_type(b[0:10])      # Revealed type is 'numpy.ndarray[builtins.float*, numpy.TwoD]'
reveal_type(b[..., 0])    # Revealed type is 'numpy.ndarray[builtins.float*, numpy.OneD]'
reveal_type(b[..., 0,0])  # Revealed type is 'builtins.float*'
reveal_type(b[:, 1])      # Revealed type is 'numpy.ndarray[builtins.float*, numpy.OneD]'
reveal_type(b[[0,1]])     # Revealed type is 'numpy.ndarray[builtins.float*, numpy.TwoD]'
reveal_type(np.zeros((3,10,10))[np.ones(3, dtype=bool)])  # Revealed type is 'numpy.ndarray[builtins.float*, numpy.ThreeD]'
reveal_type(np.zeros((3,3,3))[np.ones((3,3,3), dtype=bool)])  # Revealed type is 'numpy.ndarray[builtins.float*, numpy.OneD]'
    ''')

def test_indexing_1(mypytest):
    mypytest('''
import numpy as np
i1 = np.zeros(10, dtype='int')
a = np.zeros((3, 3, 3))
reveal_type(a[1,1,1])          # Revealed type is 'builtins.float*'
reveal_type(a[(0,1,2),])       # Revealed type is 'numpy.ndarray[builtins.float*, numpy.ThreeD]'
reveal_type(a[1,1,i1])         # Revealed type is 'numpy.ndarray[builtins.float*, numpy.OneD]'
reveal_type(a[1,i1,i1])        # Revealed type is 'numpy.ndarray[builtins.float*, numpy.OneD]'
reveal_type(a[i1,i1,i1])       # Revealed type is 'numpy.ndarray[builtins.float*, numpy.OneD]'
reveal_type(a[...])            # Revealed type is 'numpy.ndarray[builtins.float*, numpy.ThreeD]'
reveal_type(a[tuple([1,1,1])]) # Revealed type is 'numpy.ndarray[builtins.float*, Any]'
''')

def test_indexing_2(mypytest):
    mypytest('''
import numpy as np
b = np.zeros((10, 10))
reveal_type(b[None])  # Revealed type is 'numpy.ndarray[builtins.float*, numpy.ThreeD]'
reveal_type(b[np.newaxis])  # Revealed type is 'numpy.ndarray[builtins.float*, numpy.ThreeD]'
reveal_type(b[np.zeros(10, dtype='int')])  # Revealed type is 'numpy.ndarray[builtins.float*, numpy.TwoD]'
''')
