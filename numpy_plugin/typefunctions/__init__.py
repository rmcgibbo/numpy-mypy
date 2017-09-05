from .registry import registry

registry = {}


def register(name: str):
    def inner(f):
        registry[name] = f
        return f

    return inner

from . import dtype
from . import ndims
from . import ufuncs

__all__ = ['registry']
