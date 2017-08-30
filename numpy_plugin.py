import sys
import numpy as np
import functools
import itertools
from typing import Optional, Callable, List, Dict, Tuple
from collections import namedtuple, defaultdict
from mypy.types import (
    Type, Instance, CallableType, TypedDictType, UnionType, NoneTyp, FunctionLike, TypeVarType,
    AnyType, TypeList, UnboundType, TupleType, Any, TypeQuery
)
from mypy.nodes import IntExpr, TupleExpr, StrExpr, NameExpr, ListExpr, Node, MemberExpr, UnaryExpr
from mypy.plugin import Plugin, FunctionContext, AttributeContext, AnalyzeTypeContext, MethodSigContext, MethodContext
from mypy.options import Options
from mypy.sametypes import is_same_type


import logging
log = logging.getLogger(__name__)

BoundArgument = namedtuple('BoundArgument', ('name', 'formal_typ', 'arg_typ', 'arg',))

INT_TO_DIMTYPE = {
        1: 'OneD',
        2: 'TwoD',
        3: 'ThreeD'
}
DIMTYPE_TO_INT = {v: k for k, v in INT_TO_DIMTYPE.items()}


def map_name_to_args(callee, ctx, calltype='function'):
    name2arg = {}
    if calltype == 'method':
        #name2arg['self'] = BoundArgument('self', callee.arg_types[0], None, None)
        arg_types = callee.arg_types[1:]
        arg_names = callee.arg_names[1:]
    else:
        arg_types = callee.arg_types
        arg_names = callee.arg_names

    for name, formal_typ, arg_typ, arg in zip(arg_names, arg_types, ctx.arg_types, ctx.args):
        if len(arg) > 0 and len(arg_typ) > 0:
            ba = BoundArgument(name, formal_typ, arg_typ[0], arg[0])
        else:
            ba = None
        name2arg[name] = ba

    return name2arg


class HasInstanceQuery(TypeQuery[bool]):
    def __init__(self, fullname) -> None:
        super().__init__(any)
        self.fullname = fullname

    def visit_instance(self, t: Instance) -> bool:
        if t.type.fullname() == self.fullname:
            return True
        else:
            return super().visit_instance(t)


def _InferNdims(funcname: str, return_type_args, bound_args: Dict[str, BoundArgument], ctx: FunctionContext):
    dtype, ndim = return_type_args
    assert ndim.type.name() == '_InferNdims'
    st = ctx.api.modules['numpy'].names['ShapeType'].type

    matches = [f for f in bound_args.values() if f is not None and is_same_type(f.formal_typ, st)]
    if len(matches) == 1:
        ndim = infer_ndim(matches[0])

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



def _IndexHook(funcname: str, return_type_args: Tuple, bound_args: Dict[str, BoundArgument],
               ctx: FunctionContext):

    dtype, ndim = return_type_args
    self_type = ctx.type
    assert len(bound_args) == 1
    index_arg = next(iter(bound_args.values()))
    index_arg_typ = index_arg.arg_typ
    self_ndim_name = self_type.args[1].type.name()

    if self_ndim_name not in DIMTYPE_TO_INT:
        return dtype, ndim

    self_ndim_int = DIMTYPE_TO_INT[self_ndim_name]

    if isinstance(index_arg_typ, Instance):
        if index_arg_typ.type.name() == 'int':
            result_ndim = self_ndim_int - 1
        elif index_arg_typ.type.name() == 'ndarray':
            index_dtype, index_ndim = index_arg_typ.args
            if isinstance(index_dtype, Instance) and index_dtype.type.name() == 'int' and isinstance(index_ndim, Instance) and index_ndim.type.name() in DIMTYPE_TO_INT:
                # indexing with an int array of known dimension
                index_ndim_int = DIMTYPE_TO_INT[index_ndim.type.name()]
                result_ndim = self_ndim_int + index_ndim_int - 1
            elif isinstance(index_dtype, Instance) and index_dtype.type.name() == 'bool' and isinstance(index_ndim, Instance) and index_ndim.type.name() in DIMTYPE_TO_INT:
                # indexing with an bool array of known dimension
                index_ndim_int = DIMTYPE_TO_INT[index_ndim.type.name()]
                result_ndim = self_ndim_int - index_ndim_int + 1


    elif isinstance(index_arg_typ, TupleType):
        result_ndim = self_ndim_int
        for arg in index_arg_typ.items:
            if isinstance(arg, Instance) and arg.type.name() == 'int':
                result_ndim -= 1
            elif isinstance(arg, Instance) and arg.type.name() == 'slice':
                pass
            else:
                raise ValueError()


    return dtype, result_ndim






    import IPython; IPython.embed()
    raise ValueError('here')


class NumpyPlugin(Plugin):
    typefunctions  = {
        'numpy._RaiseDim': _RaiseDim,
        'numpy._InferNdims': _InferNdims,
        'numpy._InferDtype': _InferDtype,
        'numpy._InferDtypeWithDefault': _InferDtypeWithDefault,
    }
    special_ndarray_hooks = {
        'numpy.ndarray.__getitem__': _IndexHook,
    }

    def __init__(self, options: Options):
        super().__init__(options)

        self.is_setup = False
        self.api = None
        self.npmodule = None
        self.hooked_functions = {}
        self.fullname2sig = {}

    def do_setup(self, ctx: FunctionContext):
        self.api = ctx.api
        self.npmodule = ctx.api.modules['numpy']

        hooked_functions = defaultdict(list)
        for node in itertools.chain(self.npmodule.names.values(), self.npmodule.names['ndarray'].node.names.values()):
            if isinstance(node.type, CallableType):
                for tfname, tffunc in self.typefunctions.items():
                    if node.type.accept(HasInstanceQuery(tfname)):
                        hooked_functions[node.fullname].append(tffunc)
                        self.fullname2sig[node.fullname] = node.type

        for fullname, func in self.special_ndarray_hooks.items():
            hooked_functions[fullname].append(func)
            # todo: factor out this lookup
            self.fullname2sig[fullname] = self.npmodule.names['ndarray'].node.names[fullname.split('.')[-1]].type

        self.hooked_functions = dict(hooked_functions)
        self.is_setup = True


    def function_hook(self, fullname: str, calltype: str, ctx: FunctionContext):
        if not self.is_setup:
            self.do_setup(ctx)
            if fullname not in self.hooked_functions:
                return ctx.default_return_type

        assert fullname in self.hooked_functions
        return_type_args = ctx.default_return_type.args
        callee = self.fullname2sig[fullname]
        bound_args = map_name_to_args(callee, ctx, calltype=calltype)

        for tf in self.hooked_functions[fullname]:
            return_type_args = tf(funcname=fullname, return_type_args=return_type_args, bound_args=bound_args, ctx=ctx)

        dtype, ndim = return_type_args
        if ndim == 0:
            return dtype_to_named(ctx, dtype)
        else:
            return ctx.default_return_type.copy_modified(
                args=[dtype_to_named(ctx, dtype), ndim_to_named(ctx, ndim)])



    def get_function_hook(self, fullname):
        if (not self.is_setup) or fullname in self.hooked_functions:
            return functools.partial(self.function_hook, fullname,  'function')
        return False


    def get_method_hook(self, fullname: str
                        ) -> Optional[Callable[[MethodContext], Type]]:
        if (not self.is_setup) or fullname in self.hooked_functions:
            return functools.partial(self.function_hook, fullname, 'method')
        return False


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


def dtype_to_named(ctx, dtype):
    if isinstance(dtype, Type):
        return dtype

    if dtype == 'Any':
        return AnyType()

    return ctx.api.named_type(dtype)


def ndim_to_named(ctx, ndim):
    if ndim == 'Any':
        return AnyType()
    if isinstance(ndim, Type):
        return ndim

    a = {
        1: 'OneD',
        2: 'TwoD',
        3: 'ThreeD'
    }[ndim]
    return ctx.api.named_type('numpy.%s' % a)


def plugin(version):
    return NumpyPlugin