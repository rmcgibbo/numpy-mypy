from typing import Dict
import logging
from mypy.types import Type, AnyType, Instance
from mypy.nodes import IntExpr, UnaryExpr, TupleExpr, ListExpr

from .registry import register
from ..shortcuts import is_shapetype, dim_as_type, DIMTYPE_TO_INT
from ..bind_arguments import BoundArgument

log = logging.getLogger(__name__)


@register('numpy._InferNdimsFromShape')
def InferNdimsFromShape(typ: Type, funcname: str, bound_args: Dict[str, BoundArgument]):
    matches = [f for f in bound_args.values() if f is not None and is_shapetype(f.formal_typ)]
    assert len(matches) == 1
    return infer_ndim(matches[0])


@register('numpy._RaiseDim')
def RaiseDim(typ: Type, funcname: str, bound_args: Dict[str, BoundArgument]):

    arg = typ.args[0]
    if isinstance(arg, Instance):
        ndim = DIMTYPE_TO_INT[arg.type.name()] + 1
    else:
        ndim = 'Any'

    result = dim_as_type(ndim)
    return result



##############################################

def infer_ndim(formal_arg) -> str:
    arg_type = formal_arg.arg_typ
    arg = formal_arg.arg

    if isinstance(arg, IntExpr):
        ndim = 1
    elif isinstance(arg, UnaryExpr):
        ndim = 1
    elif isinstance(arg, TupleExpr):
        ndim = arg_type.length()
    elif isinstance(arg, ListExpr):
        ndim = len(arg.items)
    else:
        log.error('could not infer ndim %s %s', arg_type, arg)
        ndim = 'Any'        

    return dim_as_type(ndim)
