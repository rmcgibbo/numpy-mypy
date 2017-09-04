from typing import List
from mypy.types import (
    Type, Instance, CallableType, TypedDictType, UnionType, NoneTyp, FunctionLike, TypeVarType,
    AnyType, TypeList, UnboundType, TupleType, Any, TypeQuery, TypeVisitor
)


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

