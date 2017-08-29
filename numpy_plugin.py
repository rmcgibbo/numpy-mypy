import sys
import numpy as np
import functools
from typing import Optional, Callable
from mypy.types import (
    Type, Instance, CallableType, TypedDictType, UnionType, NoneTyp, FunctionLike, TypeVarType,
    AnyType, TypeList, UnboundType, TupleType, Any
)
from mypy.nodes import IntExpr, TupleExpr, StrExpr, NameExpr, ListExpr, Node, MemberExpr
from mypy.plugin import Plugin, FunctionContext, AttributeContext, AnalyzeTypeContext, MethodSigContext, MethodContext
import logging
log = logging.getLogger(__name__)



def map_name_to_args(funcname, ctx):
    callee = ctx.api.modules['numpy'].names[funcname].type
    name2arg = {}
    for arg_name, arg_type, arg in zip(callee.arg_names, ctx.arg_types, ctx.args):
        if len(arg_type) > 0 and len(arg) > 0:
            name2arg[arg_name] = (arg_type[0], arg[0])
        else:
            name2arg[arg_name] = None

    return name2arg


INFER_SHAPE_AND_TYPE_FUNCTIONS = (
    'numpy.zeros_like',
    'numpy.ones_like',
    'numpy.empty_like',
    'numpy.full_like',
    'numpy.zeros',
    'numpy.ones',
    'numpy.empty',
    'numpy.full')

INFER_OBJECT_AND_TYPE_FUNCTIONS = (
    'numpy.array',
    'numpy.asarray',
    'numpy.ascontiguousarray')

INFER_DTYPE_FUNCTIONS = (
    'numpy.fromstring',)



class NumpyPlugin(Plugin):
    def get_function_hook(self, fullname):
        if fullname in INFER_SHAPE_AND_TYPE_FUNCTIONS:
           name = fullname.split('.')[1]
           return functools.partial(infer_shape_and_dtype, name)
        if fullname in INFER_OBJECT_AND_TYPE_FUNCTIONS:
           name = fullname.split('.')[1]
           return functools.partial(infer_object_and_dtype, name)           
        if fullname in INFER_DTYPE_FUNCTIONS:
           name = fullname.split('.')[1]
           return functools.partial(infer_shape_and_dtype, name)
        if fullname in ('numpy.diag'):
            return diag_hook
        if fullname in ('numpy.any', 'numpy.all', 'numpy.sum'):
            name = fullname.split('.')[1]
            return functools.partial(function_hook, name)
        return False

    def get_method_hook(self, fullname: str
                        ) -> Optional[Callable[[MethodContext], Type]]:
        if fullname.startswith('numpy.ndarray'):
            methname = fullname.rsplit('.', 1)[1]
            if methname.startswith('__') and methname.endswith('__'):
                return functools.partial(magic_method_hook, methname)

        return None


def infer_shape_and_dtype(funcname, ctx):
    name2arg = map_name_to_args(funcname, ctx)
    dtype, ndim = ctx.default_return_type.args


    if 'dtype' in name2arg:
        # function signature has dtype argument
        if name2arg['dtype'] is not None:
            # user passed in a dtype argument
            dtype_str = infer_dtype(*name2arg['dtype'])
            dtype = dtype_to_named(ctx, dtype_str)
        else:
            # user passed in no dtype argument
            # we guess float
            dtype = dtype_to_named(ctx, 'float')

    if 'shape' in name2arg:
        # function signature has a 'shape' argument
        if name2arg['shape'] is not None:
            # user passed in a value
            ndim = ndim_to_named(ctx, infer_ndim(*name2arg['shape']))


    return ctx.default_return_type.copy_modified(args=[dtype, ndim])


def infer_object_and_dtype(funcname, ctx):
    name2arg = map_name_to_args(funcname, ctx)
    dtype, ndim = ctx.default_return_type.args

    if 'dtype' in name2arg:
        # function signature has dtype argument
        if name2arg['dtype'] is not None:
            # user passed in a dtype argument
            dtype_str = infer_dtype(*name2arg['dtype'])
            dtype = dtype_to_named(ctx, dtype_str)

    if 'object' in name2arg and name2arg['object'] is not None:
        arg_type, arg = name2arg['object']
        if str(arg_type) == 'builtins.list[builtins.int]':
            ndim = ndim_to_named(ctx, 1)
            dtype = dtype_to_named(ctx, 'int')
        elif str(arg_type) == 'builtins.list[builtins.int*]':
            ndim = ndim_to_named(ctx, 1)
            dtype = dtype_to_named(ctx, 'int')            
        elif str(arg_type) == 'builtins.list[Tuple[builtins.int, builtins.int]]':
            ndim = ndim_to_named(ctx, 2)
            dtype = dtype_to_named(ctx, 'int')   

    return ctx.default_return_type.copy_modified(args=[dtype, ndim])



def infer_dtype(arg_type: Type, arg: Node) -> str:
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



def infer_ndim(arg_type: Type, arg: Node) -> int:
    if isinstance(arg, IntExpr):
        ndim = 1
    elif isinstance(arg, TupleExpr):
        ndim = arg_type.length()
    elif isinstance(arg, ListExpr):
        ndim = len(arg.items)
    else:
        log.error('could not infer ndim %s %s', arg_type, arg)
        ndim = 'Any'        

    return ndim


def magic_method_hook(methname: str, ctx: MethodContext):
    # print(methname)
    try:
        self = as_dummy(ctx.type)
        arg = as_dummy(ctx.arg_types[0][0])
    except ValueError:
        print('q cant infer arg', ctx.type, ctx.arg_types[0][0])
        return ctx.default_return_type

    try:
        if methname == '__rdiv__':
            methname = '__rtruediv__'
        if methname == '__div__':
            methname = '__truediv__'
        with np.errstate(divide='ignore',invalid='ignore'):
            result = getattr(self, methname)(arg)
    except (ValueError, IndexError, TypeError):
        return ctx.default_return_type

    return to_dummy(result, ctx.api)


def diag_hook(ctx: FunctionContext):
    self = ctx.arg_types[0][0]
    if hasattr(self, 'args') and len(self.args) == 2:
        dim = self.args[1]
        if dim.type.name() == 'TwoD':
            dim = ndim_to_named(ctx, 1)
        elif dim.type.name() == 'OneD':
            dim = ndim_to_named(ctx, 2)
        return self.copy_modified(args=[self.args[0], dim])
    else:
        return ctx.default_return_type


def function_hook(funcname, ctx: MethodContext):
    try:
        arg = as_dummy(ctx.arg_types[0][0])
    except ValueError:
        # print('amy_hook: cant infer arg', ctx.arg_types[0][0])
        return ctx.default_return_type

    axis = None
    if len(ctx.arg_types[1]) > 0:
        axis = ctx.args[1][0].value

    try:        
        with np.errstate(divide='ignore', invalid='ignore'):
            result = getattr(np, funcname)(arg, axis=axis)
    except (ValueError, IndexError, TypeError):
        return ctx.default_return_type

    return to_dummy(result, ctx.api)



def as_dummy(t: Instance):

    if str(t) in ('builtins.int', 'builtins.int*'):
        return 0
    elif str(t) == 'builtins.float':
        return 0.0
    elif str(t) == 'builtins.list[builtins.int]':
        return [0]
    elif str(t) == 'builtins.list[builtins.int*]':
        return [0]
    elif str(t) == 'Tuple[builtins.int, builtins.int]':
        return (0,0)
    elif str(t) == 'builtins.slice':
        return slice(None)
    elif str(t) == 'Tuple[builtins.slice, builtins.slice]':
        return (slice(None), slice(None))
    elif str(t) == 'Tuple[builtins.slice, builtins.int]':
        return (slice(None), 0)
    elif str(t) == 'Tuple[builtins.slice, builtins.slice, builtins.int]':
        return (slice(None), slice(None), 0)
    elif str(t) == 'Tuple[builtins.int, builtins.slice]':
        return (0, slice(None))
    elif str(t) == 'numpy.ndarray[builtins.float, numpy.OneD]':
        return np.zeros(1,)
    elif str(t) == 'numpy.ndarray[builtins.float*, numpy.OneD]':        
        return np.zeros(1,)
    elif str(t) == 'numpy.ndarray[builtins.float*, numpy.OneD*]':
        return np.zeros(1,)
    elif str(t) == 'numpy.ndarray[builtins.int, numpy.OneD]':
        return np.zeros(1, dtype='i')
    elif str(t) == 'numpy.ndarray[builtins.int,* numpy.OneD]':
        return np.zeros(1, dtype='i')
    elif str(t) == 'numpy.ndarray[builtins.float, numpy.TwoD]':
        return np.zeros((1,1))
    elif str(t) == 'numpy.ndarray[builtins.float*, numpy.TwoD]':
        return np.zeros((1,1))
    elif str(t) == 'numpy.ndarray[builtins.float*, numpy.TwoD*]':
        return np.zeros((1,1))
    elif str(t) == 'numpy.ndarray[builtins.int, numpy.TwoD]':
        return np.zeros((1,1), dtype='i')
    elif str(t) == 'numpy.ndarray[builtins.int*, numpy.TwoD]':
        return np.zeros((1,1), dtype='i')
    elif str(t) == 'numpy.ndarray[builtins.bool, numpy.OneD]':
        return np.zeros(1, dtype=bool)
    elif str(t) == 'numpy.ndarray[builtins.bool, numpy.OneD*]':
        return np.zeros(1, dtype=bool)
    elif str(t) == 'numpy.ndarray[builtins.bool, numpy.TwoD]':
        return np.zeros((1,1), dtype=bool)
    elif str(t) == 'numpy.ndarray*[builtins.int, numpy.OneD]':
        return np.zeros(1, dtype='i')
    else:
        raise ValueError()


def to_dummy(v, api):
    if isinstance(v, (bool, np.bool_)):
        return api.named_type('bool')
    elif isinstance(v, (int, np.int32, np.int64)):
        return api.named_type('int')
    elif isinstance(v, float):
        return api.named_type('float')
    elif isinstance(v, np.ndarray) and v.dtype.kind == 'f' and v.ndim == 1:
        return api.named_generic_type('numpy.ndarray', [api.named_type('float'), api.named_type('numpy.OneD')])
    elif isinstance(v, np.ndarray) and v.dtype.kind == 'f' and v.ndim == 2:
        return api.named_generic_type('numpy.ndarray', [api.named_type('float'), api.named_type('numpy.TwoD')])
    elif isinstance(v, np.ndarray) and v.dtype.kind == 'i' and v.ndim == 1:
        return api.named_generic_type('numpy.ndarray', [api.named_type('int'), api.named_type('numpy.OneD')])
    elif isinstance(v, np.ndarray) and v.dtype.kind == 'i' and v.ndim == 2:
        return api.named_generic_type('numpy.ndarray', [api.named_type('int'), api.named_type('numpy.TwoD')])
    elif isinstance(v, np.ndarray) and v.dtype.kind == 'b' and v.ndim == 2:
        return api.named_generic_type('numpy.ndarray', [api.named_type('bool'), api.named_type('numpy.TwoD')])
    elif isinstance(v, np.ndarray) and v.dtype.kind == 'b' and v.ndim == 1:
        return api.named_generic_type('numpy.ndarray', [api.named_type('bool'), api.named_type('numpy.OneD')])
    else:
        print(v, file=sys.stderr)
        raise ValueError('d')


def dtype_to_named(ctx, dtype):
    if dtype == 'Any':
        return AnyType()
    return ctx.api.named_type(dtype)


def ndim_to_named(ctx, ndim):
    if ndim == 'Any':
        return AnyType()

    a = {
        1: 'OneD',
        2: 'TwoD',
        3: 'ThreeD'
    }[ndim]
    return ctx.api.named_type('numpy.%s' % a)


def plugin(version):
    return NumpyPlugin