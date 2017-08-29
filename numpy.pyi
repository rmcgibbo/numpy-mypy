"""
Numpy's mypy stub. Only type declarations for ndarray, the scalar hierarchy and array creation
methods are provided.
"""

from typing import (Any, Callable, Dict, Generic, Iterator, List, Optional, Sequence, Tuple, Type, Text,
                    TypeVar, Union, Sized, Iterable, SupportsInt, SupportsFloat, overload, SupportsAbs)

class dtype: ...
_dtype = dtype

int32 = Generic[int]


class flagsobj:
    """numpy.flagsobj"""
    aligned = None       # type: bool
    behaved = None       # type: bool
    c_contiguous = None  # type: bool
    carray = None        # type: bool
    contiguous = None    # type: bool
    f_contiguous = None  # type: bool
    farray = None        # type: bool
    fnc = None           # type: bool
    forc = None          # type: bool
    fortran = None       # type: bool
    owndata = None       # type: bool
    updateifcopy = None  # type: bool
    writeable = None     # type: bool
    def __getitem__(self, item: str) -> bool: ...
    def __setitem__(self, item: str, value: bool) -> None: ...

#
# Type variables. _T wasn't used to avoid confusions with ndarray's "T" attribute.
#

class OneD: ...
class TwoD: ...
class ThreeD: ...

_S = TypeVar('_S')
_D = TypeVar('_D', OneD, TwoD, ThreeD)


#
# Auxiliary types
#

ShapeType = Union[int, Sequence[int]]
AxesType = Union[int, Tuple[int, ...]]
OrderType = Union[str, Sequence[str]]
DtypeType = Union[dtype, type, str]
_IndexType = Union[int, List[int], slice, 'ndarray[int, Any]', 'ndarray[bool, Any]']
IndexType = Union[_IndexType, Tuple[_IndexType, ...]]

scalar = TypeVar('scalar', float, int, bool)


class flatiter(Generic[_S], Iterator[_S]):
    coords = ...  # type: ShapeType
    def copy(self) -> flatiter[_S]: ...


class ndarray(Generic[_S, _D], SupportsAbs['ndarray[_S, _D]'], Sized, Iterable):
    T = None         # type: ndarray[_S, _D]
    data = None      # type: Any
    dtype = None     # type: _dtype
    flags = None     # type: flagsobj
    flat = None      # type: flatiter[_S]
    imag = None      # type: ndarray[_S, _D]
    real = None      # type: ndarray[_S, _D]
    size = None      # type: int
    itemsize = None  # type: int
    nbytes = None    # type: int
    ndim = None      # type: int
    shape = None     # type: Tuple[int, ...]
    strides = None   # type: Tuple[int, ...]
    base = None      # type: Optional[ndarray[_S, _D]]

    # "only integers, slices (`:`), ellipsis (`...`), numpy.newaxis (`None`)
    # and integer or boolean arrays are valid indices"
    # FIXME: missing some
    def __getitem__(self, key: IndexType) -> 'ndarray[_S, Any]': ...
    def __setitem__(self, key: IndexType, value: Union[scalar, 'ndarray']) -> None: ...


    def __add__(self, value: Union[int, float, 'ndarray[Any, Any]']) -> 'ndarray[Any, Any]': ...
    def __radd__(self, value: Union[int, float, 'ndarray[Any, Any]']) -> 'ndarray[Any, Any]': ...
    def __sub__(self, value: Union[int, float, 'ndarray[Any, Any]']) -> 'ndarray[Any, Any]': ...
    def __rsub__(self, value: Union[int, float, 'ndarray[Any, Any]']) -> 'ndarray[Any, Any]': ...
    def __mul__(self, value: Union[int, float, 'ndarray[Any, Any]']) -> 'ndarray[Any, Any]': ...
    def __rmul__(self, value: Union[int, float, 'ndarray[Any, Any]']) -> 'ndarray[Any, Any]': ...
    def __div__(self, value: Union[int, float, 'ndarray[Any, Any]']) -> 'ndarray[Any, Any]': ...
    def __rdiv__(self, value: Union[int, float, 'ndarray[Any, Any]']) -> 'ndarray[Any, Any]': ...
    def __floordiv__(self, value: Union[int, float, 'ndarray[Any, Any]']) -> 'ndarray[Any, Any]': ...
    def __rfloordiv__(self, value: Union[int, float, 'ndarray[Any, Any]']) -> 'ndarray[Any, Any]': ...
    def __truediv__(self, value: Union[int, float, 'ndarray[Any, Any]']) -> 'ndarray[Any, Any]': ...
    def __rtruediv__(self, value: Union[int, float, 'ndarray[Any, Any]']) -> 'ndarray[Any, Any]': ...

    def __le__(self, value: Union[int, float, 'ndarray[Any, Any]']) -> 'ndarray[Any, Any]': ...
    def __lt__(self, value: Union[int, float, 'ndarray[Any, Any]']) -> 'ndarray[Any, Any]': ...
    def __ge__(self, value: Union[int, float, 'ndarray[Any, Any]']) -> 'ndarray[Any, Any]': ...
    def __gt__(self, value: Union[int, float, 'ndarray[Any, Any]']) -> 'ndarray[Any, Any]': ...
    def __eq__(self, value: object) -> bool: ...  # fixed in plugin


    def __contains__(self, value: Union[int, float, bool]) -> bool: ...

    def copy(self, order: str='C') -> 'ndarray[_S, _D]': ...
    # Incomplete
    def all(self, axis: AxesType=None, out:'ndarray[bool, _D]'=None, keepdims: bool=False) -> bool: ...

    def swapaxes(self, axis1: int, axis2: int) -> 'ndarray[_S, _D]': ...
    def astype(self, dtype: DtypeType) -> 'ndarray[_S, _D]': ...
    def fill(self, a: scalar) -> None: ...

    @overload
    def reshape(self, a: int) -> 'ndarray[_S, OneD]': ...
    @overload
    def reshape(self, a: Tuple[int]) -> 'ndarray[_S, OneD]': ...
    @overload
    def reshape(self, a: Tuple[int, int]) -> 'ndarray[_S, TwoD]': ...
    @overload
    def reshape(self, a: Tuple[int, int, int]) -> 'ndarray[_S, ThreeD]': ...


array1d = ndarray[_S, OneD]
array2d = ndarray[_S, TwoD]
array3d = ndarray[_S, ThreeD]
array_float = ndarray[float, _D]

float32 = float
float64 = float
int32 = int
int64 = int

array_like = TypeVar('array_like', ndarray, float, int)

# INFER_SHAPE_AND_TYPE_FUNCTIONS
def zeros(shape: ShapeType, dtype: DtypeType=float, order: str='C') -> ndarray[Any, Any]: ...
def ones(shape: ShapeType, dtype: DtypeType=float, order: str='C') -> ndarray[Any, Any]: ...
def empty(shape: ShapeType, dtype: DtypeType=float, order: str='C') -> ndarray[Any, Any]: ...
def full(shape: ShapeType, fill_value: scalar, dtype: DtypeType=float, order: str='C') -> ndarray[Any, Any]: ...
def ones_like(a: ndarray[_S, _D], dtype: DtypeType=float, order: str='K') -> ndarray[_S, _D]: ...
def zeros_like(a: ndarray[_S, _D], dtype: DtypeType=float, order: str='K') -> ndarray[_S, _D]: ...
def empty_like(a: ndarray[_S, _D], dtype: DtypeType=float, order: str='K') -> ndarray[_S, _D]: ...
def full_like(a: ndarray[_S, _D], fill_value: scalar, dtype: DtypeType=float, order: str='K') -> ndarray[_S, _D]: ...

def array(object: Iterable, dtype: DtypeType=None, copy: bool=True, order: str='K', subok: bool=False, ndim: int=0) -> ndarray[Any, Any]: ...
def asarray(object: Iterable, dtype: DtypeType=None) -> ndarray[Any, Any]: ...
def ascontiguousarray(object: Iterable, dtype: DtypeType=None) -> ndarray[Any, Any]: ...


def arange(start: int, stop: int=0, step: int=0, dtype: DtypeType=float) -> array1d[float]: ...


def vstack(tup: Iterable[ndarray[_S, OneD]]) -> ndarray[_S, TwoD]: ...
def hstack(tup: Iterable[ndarray[_S, OneD]]) -> ndarray[_S, TwoD]: ...
def dstack(tup: Iterable[ndarray[_S, TwoD]]) -> ndarray[_S, ThreeD]: ...


# Incomplete in plugin
def any(a: ndarray[_S, _D], axis: AxesType=None, out: ndarray[_S, _D]=None, keepdims: bool=False) -> bool: ...
def all(a: ndarray[_S, _D], axis: AxesType=None, out: ndarray[_S, _D]=None, keepdims: bool=False) -> bool: ...
def sum(a: ndarray[_S, _D], axis: AxesType=None, out: ndarray[_S, _D]=None, keepdims: bool=False) -> _S: ...
def mean(a: ndarray[_S, _D], axis: AxesType=None, out: ndarray[_S, _D]=None, keepdims: bool=False) -> _S: ...
def max(a: ndarray[_S, _D], axis: AxesType=None, out: ndarray[_S, _D]=None, keepdims: bool=False) -> _S: ...
def amax(a: ndarray[_S, _D], axis: AxesType=None, out: ndarray[_S, _D]=None, keepdims: bool=False) -> _S: ...
def min(a: ndarray[_S, _D], axis: AxesType=None, out: ndarray[_S, _D]=None, keepdims: bool=False) -> _S: ...
def amin(a: ndarray[_S, _D], axis: AxesType=None, out: ndarray[_S, _D]=None, keepdims: bool=False) -> _S: ...
def nansum(a: ndarray[_S, _D], axis: AxesType=None, out: ndarray[_S, _D]=None, keepdims: bool=False) -> _S: ...
def nanmean(a: ndarray[_S, _D], axis: AxesType=None, out: ndarray[_S, _D]=None, keepdims: bool=False) -> _S: ...
def cumsum(a: ndarray[_S, _D], axis: AxesType=None, out: ndarray[_S, _D]=None, keepdims: bool=False) -> ndarray[_S, _D]: ...

def isclose(a: ndarray[_S, _D], b: ndarray[_S, _D], rtol: float=1e-05, atol: float=1e-08, equal_nan: bool=False) -> ndarray[bool, _D]: ...
def allclose(a: ndarray[_S, _D], b: ndarray[_S, _D], rtol: float=1e-05, atol: float=1e-08, equal_nan: bool=False) -> bool: ...

# numpy.lib.type_check
def nan_to_num(a: ndarray[_S, _D]) -> ndarray[_S, _D]: ...

def diag(a: ndarray[_S, Any]) -> ndarray[_S, Any]: ...

# @overload
# def diag(a: array1d[float]) -> array2d[float]: ...
# @overload
# def diag(a: array2d[float]) -> array1d[float]: ...

# Incomplete
def einsum(subscripts: str, *operands: ndarray[_S, Any], out: ndarray[_S, Any]=None, dtype: DtypeType=None, order: str='K', casting: str='safe') -> ndarray[Any, Any]: ...

# Incomplete
# def ix_(*args: ndarray[_S, OneD]) -> Tuple[ndarray[_S, Any], ...]: ...
@overload
def ix_(iter1: ndarray[_S, OneD]) -> Tuple[ndarray[_S, OneD]]: ...
@overload
def ix_(iter1: ndarray[_S, OneD], iter2: ndarray[_S, OneD]) -> Tuple[ndarray[_S, OneD], ndarray[_S, OneD]]: ...


def eye(N: int, M: int=None, k: int=0, dtype: DtypeType=float) -> ndarray[float, TwoD]: ...


# Ufuncs

@overload
def abs(x: scalar) -> scalar: ...
@overload
def abs(x: ndarray[_S, _D]) -> ndarray[_S, _D]: ...

@overload
def sign(x: scalar) -> scalar: ...
@overload
def sign(x: ndarray[_S, _D]) -> ndarray[_S, _D]: ...

@overload
def sqrt(x: scalar) -> float: ...
@overload
def sqrt(x: ndarray[_S, _D]) -> ndarray[float, _D]: ...

@overload
def square(x: scalar) -> scalar: ...
@overload
def square(x: ndarray[_S, _D]) -> ndarray[_S, _D]: ...

@overload
def log(x: scalar) -> float: ...
@overload
def log(x: ndarray[_S, _D]) -> ndarray[float, _D]: ...


def isnan(x: ndarray[_S, _D]) -> ndarray[bool, _D]: ...
def isinf(x: ndarray[_S, _D]) -> ndarray[bool, _D]: ...
def isfinite(x: ndarray[_S, _D]) -> ndarray[bool, _D]: ...


@overload
def where(condition: ndarray[bool, _D]) -> Tuple[ndarray[int, OneD], ...]: ...
@overload
def where(condition: ndarray[bool, _D], x: ndarray[_S, _D], y: ndarray[_S, _D]) -> ndarray[_S, _D]: ...


def insert(arr: ndarray[_S, _D], obj: Union[int, ndarray[int, OneD]], values: Union[int, float, bool, ndarray[_S, _D]], axis: int=None) -> ndarray[_S, _D]: ...
def append(arr: ndarray[_S, _D], values: Union[int, float, bool, ndarray[_S, _D]], axis: int=None) -> ndarray[_S, _D]: ...

def fromstring(string: Text, dtype: DtypeType=float, count: int=-1, sep: Text='') -> ndarray[Any, OneD]: ...


nan = ... # type: float
inf = ... # type: float
pi = ... # type: float


class linalg:
    @staticmethod
    def cholesky(a: ndarray[Any, TwoD]) -> ndarray[float, TwoD]: ...

    @staticmethod
    def eigh(a: ndarray[Any, TwoD]) -> Tuple[ndarray[float, OneD], ndarray[float, TwoD]]: ...


class finfo:
    def __init__(self, dtype: DtypeType=None) -> None: ...
    eps = None  # type: float
    min = None  # type: float
    max = None  # type: float


class testing:
    @overload
    @staticmethod
    def assert_allclose(actual: ndarray[_S, _D], desired: ndarray[_S, _D], rtol: float=None, atol: float=None, equal_nan: bool=None, err_msg: str='', verbose: bool=False) -> None: ...

    @overload
    @staticmethod
    def assert_allclose(actual: float, desired: float, rtol: float=None, atol: float=None, equal_nan: bool=None, err_msg: str='', verbose: bool=False) -> None: ...