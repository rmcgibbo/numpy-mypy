from typing import *
import itertools
import functools
from collections import defaultdict

from mypy.options import Options
from mypy.plugin import Plugin, FunctionContext, AttributeContext, AnalyzeTypeContext, MethodSigContext, MethodContext
from mypy.types import (
    Type, Instance, CallableType, TypedDictType, UnionType, NoneTyp, FunctionLike, TypeVarType,
    AnyType, TypeList, UnboundType, TupleType, Any, TypeQuery, TypeVisitor
)

from .visitor import TypeDependenciesVisitor
from .typefunctions import registry
from .indexing import ndarray_getitem
from .ndarray_constructor import ndarray_constructor
from .bind_arguments import bind_arguments
from .ufuncs import ufunc_cast, broadcast
from . import shortcuts


class NumpyPlugin(Plugin):
    special_ndarray_hooks = {
        'numpy.ndarray.__getitem__': ndarray_getitem,
        'numpy.array': ndarray_constructor,
        'numpy.asarray': ndarray_constructor,
        'numpy.ascontiguousarray': ndarray_constructor,
    }

    def __init__(self, options: Options):
        super().__init__(options)

        self.is_setup = False
        self.api = None
        self.npmodule = None
        self.hooked_functions = set()
        self.fullname2sig = {}

    def do_setup(self, ctx: FunctionContext):
        self.api = ctx.api
        self.npmodule = ctx.api.modules['numpy']
        shortcuts.API = self.api

        for node in itertools.chain(self.npmodule.names.values(), self.npmodule.names['ndarray'].node.names.values()):
            if isinstance(node.type, CallableType):
                for tfname, tffunc in registry.items():
                    if node.type.accept(HasInstanceQuery(tfname)):
                        self.hooked_functions.add(node.fullname)
                        self.fullname2sig[node.fullname] = node.type

        for fullname, func in self.special_ndarray_hooks.items():
            self.hooked_functions.add(fullname)
            split = fullname.split('.')
            if len(split) == 2:
                assert split[0] == 'numpy'
                self.fullname2sig[fullname] = self.npmodule.names[split[1]].type
            elif len(split) == 3:
                assert split[0] == 'numpy'
                self.fullname2sig[fullname] = self.npmodule.names[split[1]].node.names[split[2]].type
            else:
                assert False

        self.is_setup = True


    def function_hook(self, fullname: str, calltype: str, ctx: FunctionContext):
        if not self.is_setup:
            self.do_setup(ctx)
            if fullname not in self.hooked_functions:
                return ctx.default_return_type

        assert fullname in self.hooked_functions
        return_type_args = ctx.default_return_type.args
        callee = self.fullname2sig[fullname]
        bound_args = bind_arguments(callee, ctx, calltype=calltype)

        if fullname in self.special_ndarray_hooks:
            return self.special_ndarray_hooks[fullname](bound_args, ctx)

        result = ctx.default_return_type.accept(TypeDependenciesVisitor(registry, fullname, bound_args))
        return result


    def get_function_hook(self, fullname):
        if (not self.is_setup) or fullname in self.hooked_functions:
            return functools.partial(self.function_hook, fullname,  'function')
        return False

    def get_method_hook(self, fullname: str
                        ) -> Optional[Callable[[MethodContext], Type]]:
        if (not self.is_setup) or fullname in self.hooked_functions:
            return functools.partial(self.function_hook, fullname, 'method')
        return False


class HasInstanceQuery(TypeQuery[bool]):
    def __init__(self, fullname) -> None:
        super().__init__(any)
        self.fullname = fullname

    def visit_instance(self, t: Instance) -> bool:
        if t.type.fullname() == self.fullname:
            return True
        else:
            return super().visit_instance(t)
