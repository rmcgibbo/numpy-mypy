from typing import *
import itertools
import functools
from collections import namedtuple, defaultdict

from mypy.options import Options
from mypy.plugin import Plugin, FunctionContext, AttributeContext, AnalyzeTypeContext, MethodSigContext, MethodContext
from mypy.types import (
    Type, Instance, CallableType, TypedDictType, UnionType, NoneTyp, FunctionLike, TypeVarType,
    AnyType, TypeList, UnboundType, TupleType, Any, TypeQuery
)

BoundArgument = namedtuple('BoundArgument', ('name', 'formal_typ', 'arg_typ', 'arg',))

from .typefunctions import (_RaiseDim, _LowerDim, _LowerDim2, _InferNdimsFromShape,
                            _InferNdimsReduction, _InferNdimsIfAxisSpecified, _InferDtype,
                            _InferDtypeWithDefault, _ToggleDims_12_21)
from .indexing import _IndexHook

from . import shortcuts


class HasInstanceQuery(TypeQuery[bool]):
    def __init__(self, fullname) -> None:
        super().__init__(any)
        self.fullname = fullname

    def visit_instance(self, t: Instance) -> bool:
        if t.type.fullname() == self.fullname:
            return True
        else:
            return super().visit_instance(t)




class NumpyPlugin(Plugin):
    typefunctions  = {
        'numpy._RaiseDim': _RaiseDim,
        'numpy._LowerDim': _LowerDim,
        'numpy._LowerDim2': _LowerDim2,
        'numpy._InferNdimsFromShape': _InferNdimsFromShape,
        'numpy._InferNdimsReduction': _InferNdimsReduction,
        'numpy._InferNdimsIfAxisSpecified': _InferNdimsIfAxisSpecified,
        'numpy._InferDtype': _InferDtype,
        'numpy._InferDtypeWithDefault': _InferDtypeWithDefault,
        'numpy._ToggleDims_12_21': _ToggleDims_12_21,

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

        s = shortcuts._NumpyShortcuts(self.api)
        for method in dir(s):
            a = getattr(s, method)
            if not method.startswith('__') and callable(a):
                setattr(shortcuts, method, a)

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
