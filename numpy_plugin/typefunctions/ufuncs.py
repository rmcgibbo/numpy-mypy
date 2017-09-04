import numpy as np
from typing import Dict
import logging
from functools import lru_cache
from mypy.types import Type, AnyType, Instance, TupleType, NoneTyp
from mypy.nodes import IntExpr, UnaryExpr, TupleExpr, ListExpr, NameExpr

from .registry import register
from ..shortcuts import is_int, is_bool, is_float, bool_type, int_type, float_type

from ..bind_arguments import BoundArgument

log = logging.getLogger(__name__)


@register('numpy._UfuncCast')
def UfuncCast(typ: Type, funcname: str, bound_args: Dict[str, BoundArgument]):
    input_chars = ''.join(map(type_to_char, typ.args))
    output_char = ufunc_to_typecodedict(funcname)[input_chars]
    
    dtype = char_to_type(output_char)
    return dtype


###############################################################################


@lru_cache()
def ufunc_to_typecodedict(funcname: str) -> Dict[str, str]:
    funcname_split = funcname.split('.')
    assert len(funcname_split) == 2 and funcname_split[0] == 'numpy'
    ufunc_name = funcname_split[1]

    types = getattr(np, ufunc_name).types
    return dict(l.split('->') for l in types)    


@lru_cache()
def type_to_char(type: Type) -> str:
    # Example: 'builtins.bool' -> '?''
    #          'builtins.float' -> 'd'
    #          'builtins.int' -> 'l'

    if is_bool(type):
        return np.dtype('bool').char
    if is_int(type):
        return np.dtype('int').char
    if is_float(type):
        return np.dtype('float').char
    raise ValueError(type)


@lru_cache()
def char_to_type(char: str) -> Type:
    if char == np.dtype('bool').char:
        return bool_type()
    elif char == np.dtype('float').char:
        return float_type()
    elif char == np.dtype('int').char:
        return int_type()
    raise ValueError(char)


