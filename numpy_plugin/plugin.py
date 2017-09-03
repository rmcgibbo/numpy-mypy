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
# from .typefunctions import (_RaiseDim, _LowerDim, _LowerDim2, _InferNdimsFromShape,
#                             _InferNdimsReduction, _InferNdimsIfAxisSpecified, _InferDtype,
#                             _InferDtypeWithDefault, _ToggleDims_12_21)
from .typefunctions import (InferDtypeWithDefault, InferNdimsFromShape)
from .indexing import ndarray_getitem
from .ndarray_constructor import ndarray_constructor
from .bind_arguments import bind_arguments
from .ufuncs import ufunc_cast, broadcast
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


def get_type_dependencies(typ: Type) -> List[str]:
    return typ.accept(TypeDependenciesVisitor())


from .shortcuts import int_type
class TypeDependenciesVisitor(TypeVisitor):
    def __init__(self, typefunctions, funcname, bound_args):
        self.typefunctions = typefunctions
        self.funcname = funcname
        self.bound_args = bound_args

    def visit_instance(self, typ: Instance) -> List[str]:
        fullname = typ.type.fullname()
        if fullname in self.typefunctions:
            return self.typefunctions[fullname](typ, self.funcname, self.bound_args)

        return Instance(typ.type, [arg.accept(self) for arg in typ.args])

    def visit_any(self, typ) -> List[str]:
        return typ

    def visit_none_type(self, typ) -> List[str]:
        return typ

    def visit_callable_type(self, typ) -> List[str]:
        raise NotImplementedError

    def visit_deleted_type(self, typ) -> List[str]:
        return typ

    def visit_partial_type(self, typ) -> List[str]:
        raise NotImplementedError

    def visit_tuple_type(self, typ) -> List[str]:
        raise NotImplementedError

    def visit_type_type(self, typ) -> List[str]:
        # TODO: replace with actual implementation
        return typ

    def visit_type_var(self, typ) -> List[str]:
        return typ

    def visit_typeddict_type(self, typ) -> List[str]:
        raise NotImplementedError

    def visit_unbound_type(self, typ) -> List[str]:
        raise NotImplementedError

    def visit_uninhabited_type(self, typ) -> List[str]:
        raise NotImplementedError

    def visit_union_type(self, typ) -> List[str]:
        return typ


class NumpyPlugin(Plugin):
    typefunctions  = {
        # 'numpy._RaiseDim': _RaiseDim,
        # 'numpy._LowerDim': _LowerDim,
        # 'numpy._LowerDim2': _LowerDim2,
        'numpy._InferNdimsFromShape': InferNdimsFromShape,
        # 'numpy._InferNdimsReduction': _InferNdimsReduction,
        # 'numpy._InferNdimsIfAxisSpecified': _InferNdimsIfAxisSpecified,
        # 'numpy._InferDtype': _InferDtype,
        'numpy._InferDtypeWithDefault': InferDtypeWithDefault,
        # 'numpy._ToggleDims_12_21': _ToggleDims_12_21,
        # 'numpy._UfuncCast': ufunc_cast,
        # 'numpy._Broadcast': broadcast,
    }

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
                for tfname, tffunc in self.typefunctions.items():
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

        result = ctx.default_return_type.accept(TypeDependenciesVisitor(self.typefunctions, fullname, bound_args))
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

