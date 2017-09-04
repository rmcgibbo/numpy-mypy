import numpy as np
import logging
from typing import Dict
from mypy.sametypes import is_same_type
from mypy.types import Type
from mypy.nodes import NameExpr, StrExpr, MemberExpr

from .registry import register
from ..shortcuts import is_dtypetype, int_type, bool_type, float_type
from ..bind_arguments import BoundArgument

log = logging.getLogger(__name__)


@register('numpy._InferDtypeWithDefault')
def InferDtypeWithDefault(typ: Type, funcname: str, bound_args: Dict[str, BoundArgument]):
    matches = [f for f in bound_args.values() if f is not None and is_dtypetype(f.formal_typ)]
    if len(matches) == 0:
        return typ.args[0]
    elif len(matches) == 1:
        return infer_dtype(matches[0])
    raise ValueError()


def infer_dtype(formal_arg) -> str:
    arg_type = formal_arg.arg_typ
    arg = formal_arg.arg
    if isinstance(arg, NameExpr):
        # np.zeros(1, int)
        dtype_str = arg.name
    elif isinstance(arg, StrExpr):
        # np.zeros(1, 'int')
        dtype_str = arg.value
    elif isinstance(arg, MemberExpr):
         # np.zeros(1, np.int32)
        dtype_str = arg.name
    else:
        log.error('could not infer dtype %s %s', arg_type, arg)
        raise ValueError()

        return 'Any'
    np_kind = np.dtype(dtype_str).kind
    builtin_kind_str = {
        'i': int_type(),
        'b': bool_type(),
        'f': float_type(),
    }[np_kind]
    return builtin_kind_str

