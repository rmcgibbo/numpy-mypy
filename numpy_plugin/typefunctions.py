import numpy as np
import logging
from typing import Dict, Tuple, Union
from mypy.plugin import FunctionContext
from mypy.sametypes import is_same_type
from mypy.subtypes import is_subtype
from mypy.nodes import IntExpr, TupleExpr, StrExpr, NameExpr, ListExpr, Node, MemberExpr, UnaryExpr, EllipsisExpr
from mypy.types import (
    Type, Instance, CallableType, TypedDictType, UnionType, NoneTyp, FunctionLike, TypeVarType,
    AnyType, TypeList, UnboundType, TupleType, Any, TypeQuery
)
from .bind_arguments import BoundArgument

log = logging.getLogger(__name__)

INT_TO_DIMTYPE = {
        1: 'OneD',
        2: 'TwoD',
        3: 'ThreeD'
}
DIMTYPE_TO_INT = {v: k for k, v in INT_TO_DIMTYPE.items()}


def _InferNdimsFromShape(funcname: str, return_type_args, bound_args: Dict[str, BoundArgument], ctx: FunctionContext):
    from .shortcuts import is_shapetype

    dtype, ndim = return_type_args
    assert ndim.type.name() == '_InferNdimsFromShape'

    matches_ShapeType = [f for f in bound_args.values() if f is not None and is_shapetype(f.formal_typ)]
    assert len(matches_ShapeType) == 1

    m = matches_ShapeType[0]
    ndim = infer_ndim(m)
    return dtype, ndim


def _InferNdimsReduction(funcname: str, return_type_args, bound_args: Dict[str, BoundArgument], ctx: FunctionContext):
    from .shortcuts import is_axestype  

    dtype, ndim = return_type_args
    assert ndim.type.name() == '_InferNdimsReduction'

    matches_AxesType = [f for f in bound_args.values() if f is not None and is_axestype(f.formal_typ)]

    if len(matches_AxesType) == 0:
        return dtype, 0

    assert len(matches_AxesType) == 1

    arg = ndim.args[0]
    if isinstance(arg, Instance):
        operand_ndim = DIMTYPE_TO_INT[arg.type.name()]
    else:
        return dtype, AnyType()
    
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
        m = matches_AxesType[0]
        if isinstance(m.arg_typ, Instance) and m.arg_typ.type.name() == 'int':
            # called with axis: int
            ndim = operand_ndim - 1
        elif isinstance(m.arg_typ, TupleType):
            ndim = operand_ndim - m.arg_typ.length()
        else:
            raise RuntimeError()
    else:
        ndim = 'Any'

    return dtype, ndim


def _InferNdimsIfAxisSpecified(funcname: str, return_type_args, bound_args: Dict[str, BoundArgument], ctx: FunctionContext):
    # if axis is None, the default, we return the first type argument
    # if axis is an int we return the second type argument
    dtype, ndim = return_type_args
    assert ndim.type.name() == '_InferNdimsIfAxisSpecified'
    assert 'axis' in bound_args
    if bound_args['axis'] is None or isinstance(bound_args['axis'].arg_typ, NoneTyp):
        # unspecified or specified as None
        ndim = ndim.args[0]
    elif isinstance(bound_args['axis'].arg_typ, Instance) and bound_args['axis'].arg_typ.type.name() == 'int':
        # passed axis: int
        ndim = ndim.args[1]
    else:
        logging.error('cannot infer')

    return dtype, ndim



def _RaiseDim(funcname: str, return_type_args: Tuple, bound_args: Dict[str, BoundArgument], ctx: FunctionContext):
    dtype, ndim = return_type_args
    assert ndim.type.name() == '_RaiseDim'

    arg = ndim.args[0]
    if isinstance(arg, Instance):
        ndim = DIMTYPE_TO_INT[arg.type.name()] + 1
    else:
        ndim = AnyType()

    return dtype, ndim

def _LowerDim(funcname: str, return_type_args: Tuple, bound_args: Dict[str, BoundArgument], ctx: FunctionContext):
    dtype, ndim = return_type_args
    assert ndim.type.name() == '_LowerDim'

    arg = ndim.args[0]
    if isinstance(arg, Instance):
        ndim = DIMTYPE_TO_INT[arg.type.name()] - 1
    else:
        ndim = AnyType()

    return dtype, ndim


def _LowerDim2(funcname: str, return_type_args: Tuple, bound_args: Dict[str, BoundArgument], ctx: FunctionContext):
    dtype, ndim = return_type_args
    assert ndim.type.name() == '_LowerDim2'

    arg = ndim.args[0]
    if isinstance(arg, Instance):
        ndim = DIMTYPE_TO_INT[arg.type.name()] - 2
    else:
        ndim = AnyType()

    return dtype, ndim


def _LowerDimIfAxisSpecified(funcname: str, return_type_args: Tuple, bound_args: Dict[str, BoundArgument], ctx: FunctionContext):
    type, ndim = return_type_args
    assert ndim.type.name() == '_LowerDimIfAxisSpecified'

    arg = ndim.args[0]
    if isinstance(arg, Instance) :
        ndim = DIMTYPE_TO_INT[arg.type.name()] - 1
    else:
        ndim = AnyType()

    return dtype, ndim


def _ToggleDims_12_21(funcname: str, return_type_args, bound_args: Dict[str, BoundArgument], ctx: FunctionContext):
    dtype, ndim = return_type_args
    assert ndim.type.name() == '_ToggleDims_12_21'

    arg = ndim.args[0]
    if isinstance(arg, Instance):
        input_ndim = DIMTYPE_TO_INT[arg.type.name()]
        if input_ndim == 2:
            ndim = 1
        elif input_ndim == 1:
            ndim = 2
        else:
            raise RuntimeError()
    else:
        ndim = AnyType()

    return dtype, ndim

def _InferDtype(funcname: str, return_type_args: Tuple, bound_args: Dict[str, BoundArgument], ctx: FunctionContext):
    dtype, ndim = return_type_args
    assert dtype.type.name() == '_InferDtype'
    dt = ctx.api.modules['numpy'].names['DtypeType'].type
    matches = [f for f in bound_args.values() if f is not None and is_same_type(f.formal_typ, dt)]
    if len(matches) == 1:
        dtype = infer_dtype(matches[0])

    return dtype, ndim


def _InferDtypeWithDefault(funcname: str, return_type_args: Tuple, bound_args: Dict[str, BoundArgument],
                           ctx: FunctionContext):
    dtype, ndim = return_type_args
    assert dtype.type.name() == '_InferDtypeWithDefault'
    dt = ctx.api.modules['numpy'].names['DtypeType'].type
    matches = [f for f in bound_args.values() if f is not None and is_same_type(f.formal_typ, dt)]

    if len(matches) == 0:
        dtype = dtype.args[0]
    elif len(matches) == 1:
        dtype = infer_dtype(matches[0])

    return dtype, ndim


 ######################################################


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

    return ndim


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
        'i': 'int',
        'b': 'bool',
        'f': 'float',
    }[np_kind]
    return builtin_kind_str

