from functools import lru_cache
from mypy.types import NoneTyp, TupleType, UnionType, AnyType, Type
from mypy.sametypes import is_same_type
from mypy.subtypes import is_subtype
from .typefunctions import DIMTYPE_TO_INT

API = None

@lru_cache()
def is_int(type: Type):
    return is_same_type(type, API.named_type('builtins.int'))

@lru_cache()
def is_none(type: Type):
    return is_same_type(type, NoneTyp())

@lru_cache()
def is_object(type: Type):
    return is_same_type(type, API.named_type('builtins.object'))

@lru_cache()
def is_ellipsis(type: Type):
    return is_same_type(type, API.named_type('builtins.ellipsis'))

@lru_cache()
def is_slice(type: Type):
    return is_same_type(type, API.named_type('builtins.slice'))

@lru_cache()
def is_list_of_int(type: Type):
    return is_same_type(
        type, API.named_generic_type('builtins.list', args=[API.named_type('builtins.int')]))

@lru_cache()
def is_basic_index_sequence(type: Type):
    # From: https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html
    # >> Basic slicing occurs when obj is a slice object
    # >> an integer, o a tuple of slice objects and integers.
    # >> Ellipsis and newaxis objects can be interspersed with these as well.
    u = UnionType.make_union([
        API.named_type('builtins.int'),
        API.named_type('builtins.slice'),
        NoneTyp(),
        API.named_type('builtins.ellipsis'),
    ])
    return is_subtype(type,
                      API.named_generic_type(
                          'typing.Sequence', args=[u]))

@lru_cache()
def is_shapetype(type: Type):
    return is_same_type(type,
                        API.modules['numpy'].names['ShapeType'].type)

@lru_cache()
def is_axestype(type: Type):
    return is_same_type(type,
                        API.modules['numpy'].names['AxesType'].type)

@lru_cache()
def is_ndarray(type: Type):
    return is_subtype(type, API.named_generic_type('numpy.ndarray',
        args=[AnyType(), AnyType()]))


@lru_cache()
def is_ndarray_of_ints(type: Type, no_bools: bool=True):
    of_ints = is_subtype(type, API.named_generic_type('numpy.ndarray',
        args=[API.named_type('builtins.int'),
              API.named_type('numpy._Dimension')]))

    if not no_bools:
        return of_ints
    return of_ints and not is_ndarray_of_bools(type)

@lru_cache()
def is_ndarray_of_bools(type: Type):
    return is_subtype(type, API.named_generic_type('numpy.ndarray',
        args=[API.named_type('builtins.bool'),
              API.named_type('numpy._Dimension')]))

@lru_cache()
def ndarray_dim_as_int(type: Type):
    return DIMTYPE_TO_INT[type.args[1].type.name()]

@lru_cache()
def is_ndsequence_of(type: Type, base_type_str: str):
    i = API.named_type(base_type_str)
    si = API.named_generic_type('typing.Sequence', args=[i])
    ssi = API.named_generic_type('typing.Sequence', args=[si])
    sssi = API.named_generic_type('typing.Sequence', args=[ssi])
    u = UnionType.make_union([si, ssi, sssi])
    return is_subtype(type, u)

@lru_cache()
def is_ndsequence_of_bools(type: Type):
    return is_ndsequence_of(type, 'builtins.bool')

@lru_cache()
def is_ndsequence_of_float(type: Type):
    return is_ndsequence_of(type, 'builtins.float')


@lru_cache()
def is_ndsequence_of_ints(type: Type, no_bools: bool=True):
    of_ints = is_ndsequence_of(type, 'builtins.int')
    if not no_bools:
        return of_ints
    return of_ints and not is_ndsequence_of_bools(type)


@lru_cache()
def ndsequence_dim_as_int(type: Type):
    i = AnyType()
    si = API.named_generic_type('typing.Sequence', args=[i])
    ssi = API.named_generic_type('typing.Sequence', args=[si])
    sssi = API.named_generic_type('typing.Sequence', args=[ssi])

    if is_subtype(type, sssi):
        return 3
    elif is_subtype(type, ssi):
        return 2
    elif is_subtype(type, si):
        return 1
    raise KeyError()
