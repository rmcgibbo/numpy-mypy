from typing import Dict
import logging
from mypy.types import Type, AnyType, Instance, TupleType, NoneTyp
from mypy.nodes import IntExpr, UnaryExpr, TupleExpr, ListExpr, NameExpr

from .registry import register
from ..shortcuts import is_shapetype, is_axestype, is_int, is_tuple, dim_as_type, DIMTYPE_TO_INT
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


@register('numpy._InferNdimsReduction')
def InferNdimsReduction(typ: Type, funcname: str, bound_args: Dict[str, BoundArgument]):
    matches = [f for f in bound_args.values() if f is not None and is_axestype(f.formal_typ)]

    if len(matches) == 0:
        return dim_as_type(0)

    assert len(matches) == 1

    arg = typ.args[0]
    if isinstance(arg, Instance):
        operand_ndim = DIMTYPE_TO_INT[arg.type.name()]
    else:
        return dim_as_type('Any')

    if 'keepdims' in bound_args and bound_args['keepdims'] is not None:
        kdarg = bound_args['keepdims'].arg
        if isinstance(kdarg, NameExpr):
            if kdarg.fullname == 'builtins.True':
                keepdims = True
            elif kdarg.fullname == 'builtins.False': 
                keepdims = False
            else:
                keepdims = '?'
        else:
            keepdims = '?'
    else:
        keepdims = False


    if keepdims is True:
        ndim = operand_ndim
    elif keepdims is False:
        m = matches[0]
        if is_int(m.arg_typ):
            # called with axis: int
            ndim = operand_ndim - 1
        elif is_tuple(m.arg_typ):
            ndim = operand_ndim - m.arg_typ.length()
        else:
            raise RuntimeError()
    else:
        ndim = 'Any'

    return dim_as_type(ndim)


@register('numpy._InferNdimsIfAxisSpecified')
def InferNdimsIfAxisSpecified(typ: Type, funcname: str, bound_args: Dict[str, BoundArgument]):
    # if axis is None, the default, we return the first type argument
    # if axis is an int we return the second type argument
    assert 'axis' in bound_args
    if bound_args['axis'] is None or isinstance(bound_args['axis'].arg_typ, NoneTyp):
        # unspecified or specified as None
        ndim = typ.args[0]
    elif is_int(bound_args['axis'].arg_typ):
        # passed axis: int
        ndim = typ.args[1]
    else:
        log.error('cannot infer')

    return ndim


@register('numpy._RaiseDim')
def RaiseDim(typ: Type, funcname: str, bound_args: Dict[str, BoundArgument]):
    arg = typ.args[0]
    if isinstance(arg, Instance):
        ndim = dim_as_type(DIMTYPE_TO_INT[arg.type.name()] + 1)
    else:
        ndim = AnyType()

    return ndim

@register('numpy._LowerDim')
def LowerDim(typ: Type, funcname: str, bound_args: Dict[str, BoundArgument]):
    arg = typ.args[0]
    if isinstance(arg, Instance):
        ndim = dim_as_type(DIMTYPE_TO_INT[arg.type.name()] - 1)
    else:
        ndim = AnyType()

    return ndim

@register('numpy._LowerDim2')
def LowerDim2(typ: Type, funcname: str, bound_args: Dict[str, BoundArgument]):
    arg = typ.args[0]
    if isinstance(arg, Instance):
        ndim = dim_as_type(DIMTYPE_TO_INT[arg.type.name()] - 2)
    else:
        ndim = AnyType()

    return ndim


@register('numpy._ToggleDims_12_21')
def ToggleDims_12_21(typ: Type, funcname: str, bound_args: Dict[str, BoundArgument]):
    arg = typ.args[0]
    if isinstance(arg, Instance):
        input_ndim = DIMTYPE_TO_INT[arg.type.name()]
        if input_ndim == 2:
            ndim = dim_as_type(1)
        elif input_ndim == 1:
            ndim = dim_as_type(2)
        else:
            raise RuntimeError()
    else:
        ndim = AnyType()

    return ndim


@register('numpy._LargestDim')
def LargestDim(typ: Type, funcname: str, bound_args: Dict[str, BoundArgument]):
    arg0 = typ.args[0]
    arg1 = typ.args[1]

    if isinstance(arg0, Instance) and isinstance(arg1, Instance):
        ndim = max(DIMTYPE_TO_INT[arg0.type.name()], DIMTYPE_TO_INT[arg1.type.name()])
    else:
        ndim = 'Any'

    return dim_as_type(ndim)


###############################################################################


def infer_ndim(formal_arg) -> Type:
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
