from mypy.types import Instance, TypeVisitor, TupleType


class TypeTransformer(TypeVisitor):
    def visit_instance(self, typ: Instance):
        return typ

    def visit_any(self, typ):
        return typ

    def visit_none_type(self, typ):
        return typ

    def visit_callable_type(self, typ):
        raise NotImplementedError

    def visit_deleted_type(self, typ):
        return typ

    def visit_partial_type(self, typ):
        raise NotImplementedError

    def visit_tuple_type(self, typ):
        return TupleType([i.accept(self) for i in typ.items], typ.fallback)

    def visit_type_type(self, typ):
        # TODO: replace with actual implementation
        return typ

    def visit_type_var(self, typ):
        return typ

    def visit_typeddict_type(self, typ):
        raise NotImplementedError

    def visit_unbound_type(self, typ):
        raise NotImplementedError

    def visit_uninhabited_type(self, typ):
        raise NotImplementedError

    def visit_union_type(self, typ):
        raise NotImplementedError



class TypefunctionRegistryTransformer(TypeTransformer):
    def __init__(self, registry, funcname, bound_args):
        self.registry = registry
        self.funcname = funcname
        self.bound_args = bound_args

    def visit_instance(self, typ: Instance):
        fullname = typ.type.fullname()
        if fullname in self.registry:
            return self.registry[fullname](typ, self.funcname, self.bound_args)

        return Instance(typ.type, [arg.accept(self) for arg in typ.args])


class SimpleTransformer(TypeTransformer):
    def __init__(self, instance_function):
        self.instance_function = instance_function

    def visit_instance(self, typ: Instance):
        return self.instance_function(typ)
