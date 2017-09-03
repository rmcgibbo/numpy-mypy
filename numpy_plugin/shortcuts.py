from functools import lru_cache
from mypy.types import NoneTyp, TupleType, UnionType, AnyType
from mypy.sametypes import is_same_type
from mypy.subtypes import is_subtype


class _NumpyShortcuts(object):
    def __init__(self, api):
        self.api = api

    @lru_cache()
    def is_int(self, type):
        return is_same_type(type, self.api.named_type('builtins.int'))

    @lru_cache()
    def is_none(self, type):
        return is_same_type(type, NoneTyp())

    @lru_cache()
    def is_object(self, type):
        return is_same_type(type, self.api.named_type('builtins.object'))

    @lru_cache()
    def is_ellipsis(self, type):
        return is_same_type(type, self.api.named_type('builtins.ellipsis'))

    @lru_cache()
    def is_slice(self, type):
        return is_same_type(type, self.api.named_type('builtins.slice'))

    @lru_cache()
    def is_list_of_int(self, type):
        return is_same_type(type,
                            self.api.named_generic_type(
                                'builtins.list',
                                args=[self.api.named_type('builtins.int')]))

    @lru_cache()
    def is_basic_index_sequence(self, type):
        # From: https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html
        # >> Basic slicing occurs when obj is a slice object
        # >> an integer, or a tuple of slice objects and integers.
        # >> Ellipsis and newaxis objects can be interspersed with these as well.
        u = UnionType.make_union([
            self.api.named_type('builtins.int'),
            self.api.named_type('builtins.slice'),
            NoneTyp(),
            self.api.named_type('builtins.ellipsis'),
        ])
        return is_subtype(type,
                          self.api.named_generic_type(
                              'typing.Sequence', args=[u]))

    @lru_cache()
    def is_shapetype(self, type):
        return is_same_type(type,
                            self.api.modules['numpy'].names['ShapeType'].type)

    @lru_cache()
    def is_axestype(self, type):
        return is_same_type(type,
                            self.api.modules['numpy'].names['AxesType'].type)

    @lru_cache()
    def is_ndarray_of_ints(self, type, no_bools=True):
        of_ints = is_subtype(type,
                             self.api.named_generic_type(
                                 'numpy.ndarray',
                                 args=[
                                     self.api.named_type('builtins.int'),
                                     self.api.named_type('numpy._Dimension')
                                 ]))
        if not no_bools:
            return of_ints
        else:
            of_bools = self.is_ndarray_of_bools(type)
            return of_ints and not of_bools

    @lru_cache()
    def is_ndarray_of_bools(self, type):
        return is_subtype(type,
                          self.api.named_generic_type(
                              'numpy.ndarray',
                              args=[
                                  self.api.named_type('builtins.bool'),
                                  self.api.named_type('numpy._Dimension')
                              ]))

    @lru_cache()
    def ndarray_dim_as_int(self, type):
        from .typefunctions import DIMTYPE_TO_INT
        return DIMTYPE_TO_INT[type.args[1].type.name()]

    @lru_cache()
    def is_ndsequence_of_ints(self, type, no_bools=True):
        i = self.api.named_type('builtins.int')
        si = self.api.named_generic_type('typing.Sequence', args=[i])
        ssi = self.api.named_generic_type('typing.Sequence', args=[si])
        sssi = self.api.named_generic_type('typing.Sequence', args=[ssi])
        u = UnionType.make_union([si, ssi, sssi])
        return is_subtype(type, u)

    @lru_cache()
    def is_ndsequence_of_bools(self, type):
        i = self.api.named_type('builtins.bool')
        si = self.api.named_generic_type('typing.Sequence', args=[i])
        ssi = self.api.named_generic_type('typing.Sequence', args=[si])
        sssi = self.api.named_generic_type('typing.Sequence', args=[ssi])
        u = UnionType.make_union([si, ssi, sssi])
        return is_subtype(type, u)

    @lru_cache()
    def ndsequence_dim_as_int(self, type):
        i = AnyType()
        si = self.api.named_generic_type('typing.Sequence', args=[i])
        ssi = self.api.named_generic_type('typing.Sequence', args=[si])
        sssi = self.api.named_generic_type('typing.Sequence', args=[ssi])

        if is_subtype(type, sssi):
            return 3
        elif is_subtype(type, ssi):
            return 2
        elif is_subtype(type, si):
            return 1
        raise KeyError()
