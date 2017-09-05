import numpy as np
from typing import Dict, List, Tuple
import logging
from functools import lru_cache
from mypy.types import Type

from . import register
from ..shortcuts import (is_int, is_bool, is_float, bool_type, int_type,
                         float_type, is_ndarray_of_bools, is_ndarray_of_floats,
                         is_ndarray_of_ints)

from ..bind_arguments import BoundArgument

log = logging.getLogger(__name__)


@register('numpy._UfuncCast')
def UfuncCast(typ: Type, funcname: str, bound_args: Dict[str, BoundArgument]):
    keys = (k for k in bound_args.keys() if k not in ('out', 'out1', 'out2'))
    input_chars = tuple((type_to_char(bound_args[xx].arg_typ) for xx in keys))
    typecodes = ufunc_to_typecodes(funcname)
    output_char = ufunc_type_resolver(input_chars, typecodes)

    dtype = char_to_type(output_char[0])
    return dtype


###############################################################################


@lru_cache()
def ufunc_to_typecodes(funcname: str) -> List[Tuple[str, str]]:
    funcname_split = funcname.split('.')
    assert len(funcname_split) == 2 and funcname_split[0] == 'numpy'
    ufunc_name = funcname_split[1]

    types = getattr(np, ufunc_name).types
    return [l.split('->') for l in types]


def ufunc_type_resolver(ichars: str, typecodes: List[Tuple[str, str]]):
    for inp, out in typecodes:
        if all(np.can_cast(ii, tt) for ii, tt in zip(ichars, inp)):
            return out
    raise ValueError()


@lru_cache()
def type_to_char(type: Type) -> str:
    # Example: 'builtins.bool' -> '?''
    #          'builtins.float' -> 'd'
    #          'builtins.int' -> 'l'
    if is_bool(type) or is_ndarray_of_bools(type):
        return np.dtype('bool').char
    if is_float(type) or is_ndarray_of_floats(type):
        return np.dtype('float').char
    if is_int(type) or is_ndarray_of_ints(type):
        return np.dtype('int').char

    raise ValueError(type)


@lru_cache()
def char_to_type(char: str) -> Type:
    if char in ('b', '?'):
        return bool_type()
    elif char in ('e', 'f', 'd', 'g'):
        return float_type()
    elif char in ('i', 'l'):
        return int_type()
    raise ValueError(char)
