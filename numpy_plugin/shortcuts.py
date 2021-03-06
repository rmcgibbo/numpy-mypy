from typing import Union
from functools import lru_cache
from mypy.types import NoneTyp, UnionType, AnyType, Type, TupleType, UninhabitedType, Instance, TypeOfAny
from mypy.sametypes import is_same_type
from mypy.subtypes import is_subtype

API = None


INT_TO_DIMTYPE = {
        0: 'ZeroD',
        1: 'OneD',
        2: 'TwoD',
        3: 'ThreeD',
        4: 'FourD',
}
DIMTYPE_TO_INT = {v: k for k, v in INT_TO_DIMTYPE.items()}


@lru_cache()
def int_type():
    return API.named_type('builtins.int')


@lru_cache()
def float_type():
    return API.named_type('builtins.float')


@lru_cache()
def bool_type():
    return API.named_type('builtins.bool')


@lru_cache()
def is_int(typ: Type):
    return is_same_type(typ, int_type())


@lru_cache()
def is_bool(typ: Type):
    return is_same_type(typ, bool_type())


@lru_cache()
def is_tuple(typ: Type) -> bool:
    return isinstance(typ, TupleType)


@lru_cache()
def is_float(typ: Type):
    return is_same_type(typ, float_type())


@lru_cache()
def is_none(typ: Type):
    return is_same_type(typ, NoneTyp())


@lru_cache()
def is_object(typ: Type):
    return is_same_type(typ, API.named_type('builtins.object'))


@lru_cache()
def is_ellipsis(type: Type):
    return is_same_type(type, API.named_type('builtins.ellipsis'))


@lru_cache()
def is_slice(type: Type):
    return is_same_type(type, API.named_type('builtins.slice'))


@lru_cache()
def is_list_of_int(type: Type):
    return is_same_type(type, API.named_generic_type('builtins.list', args=[int_type()]))


@lru_cache()
def is_basic_index_sequence(type: Type):
    # From: https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html
    # >> Basic slicing occurs when obj is a slice object
    # >> an integer, o a tuple of slice objects and integers.
    # >> Ellipsis and newaxis objects can be interspersed with these as well.
    u = UnionType.make_union([
        int_type(),
        API.named_type('builtins.slice'),
        NoneTyp(),
        API.named_type('builtins.ellipsis'),
    ])
    return is_subtype(type, API.named_generic_type('typing.Sequence', args=[u]))


@lru_cache()
def is_any(typ: Type):
    return isinstance(typ, AnyType)

@lru_cache()
def is_shapetype(type: Type):
    return is_same_type(type, API.modules['numpy'].names['ShapeType'].type)


@lru_cache()
def is_axestype(type: Type):
    return is_same_type(type, API.modules['numpy'].names['AxesType'].type)


@lru_cache()
def is_dtypetype(type: Type):
    return is_same_type(type, API.modules['numpy'].names['DtypeType'].type)


@lru_cache()
def is_ndarray(typ: Type):
    if not isinstance(typ, Instance):
        return False
    return is_subtype(typ, API.named_generic_type('numpy.ndarray',
        args=[AnyType(TypeOfAny.unannotated), AnyType(TypeOfAny.unannotated)]))


@lru_cache()
def is_ndarray_of_ints(typ: Type, no_bools: bool=True):
    if not isinstance(typ, Instance):
        return False
    of_ints = is_subtype(typ, API.named_generic_type('numpy.ndarray',
        args=[int_type(), API.named_type('numpy._Dimension')])) and (not is_same_type(typ.args[1], AnyType(TypeOfAny.unannotated)))

    if not no_bools:
        return of_ints
    return of_ints and not is_ndarray_of_bools(typ)


@lru_cache()
def is_ndarray_of_bools(typ: Type):
    if not isinstance(typ, Instance):
        return False
    return is_subtype(typ, API.named_generic_type('numpy.ndarray',
        args=[bool_type(), API.named_type('numpy._Dimension')])) and (not is_same_type(typ.args[1], AnyType(TypeOfAny.unannotated)))


@lru_cache()
def is_ndarray_of_floats(typ: Type):
    if not isinstance(typ, Instance):
        return False
    return is_subtype(typ, API.named_generic_type('numpy.ndarray',
        args=[float_type(), API.named_type('numpy._Dimension')])) and (not is_same_type(typ.args[1], AnyType(TypeOfAny.unannotated)))


@lru_cache()
def ndarray_dim_as_int(type: Type) -> Union[int, str]:
    return dimtype_to_int(type.args[1]) 


@lru_cache()
def is_ndsequence_of(type: Type, base_type: Type):
    si = API.named_generic_type('typing.Sequence', args=[base_type])
    ssi = API.named_generic_type('typing.Sequence', args=[si])
    sssi = API.named_generic_type('typing.Sequence', args=[ssi])
    u = UnionType.make_union([si, ssi, sssi])
    return is_subtype(type, u)


@lru_cache()
def is_ndsequence_of_bools(type: Type):
    return is_ndsequence_of(type, bool_type()) and not isinstance(type.args[0], AnyType)


@lru_cache()
def is_ndsequence_of_floats(type: Type):
    return is_ndsequence_of(type, float_type()) and not isinstance(type.args[0], AnyType)


@lru_cache()
def is_ndsequence_of_ints(type: Type, no_bools: bool=True):
    of_ints = is_ndsequence_of(type, int_type()) and not isinstance(type.args[0], AnyType)
    if not no_bools:
        return of_ints
    return of_ints and not is_ndsequence_of_bools(type)


@lru_cache()
def ndsequence_dim_as_int(type: Type) -> int:
    i = AnyType(TypeOfAny.unannotated)
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


@lru_cache()
def ndsequence_dim_as_type(type: Type):
    return dim_as_type(ndsequence_dim_as_int(type))


@lru_cache()
def zerodim_to_scalar(typ: Type) -> Type:
    if is_subtype(typ,  API.named_generic_type('numpy.ndarray',
        args=[AnyType(TypeOfAny.unannotated), API.named_type('numpy.ZeroD')])) and (not is_same_type(typ.args[1], AnyType(TypeOfAny.unannotated))):
        return typ.args[0]
    return typ


@lru_cache()
def dimtype_to_int(typ) -> Union[int, str]:
    if isinstance(typ, Instance):
        return DIMTYPE_TO_INT[typ.type.name()]
    elif isinstance(typ, UninhabitedType):
        return 0
    else:
        return 'Any'
    raise ValueError()


@lru_cache()
def dim_as_type(i: Union[str, int, Type]):
    if isinstance(i, str):
        assert i == 'Any'
        return AnyType(TypeOfAny.unannotated)
    if i in INT_TO_DIMTYPE:
        return  API.named_type('numpy.%s' % INT_TO_DIMTYPE[i])

    raise ValueError(i, type(i))
