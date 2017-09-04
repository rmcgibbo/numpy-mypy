from typing import Union, Tuple, Dict
from mypy.types import Type, TupleType
from mypy.plugin import FunctionContext
from itertools import chain
from .bind_arguments import BoundArgument
from .shortcuts import (is_int, is_ndarray_of_ints, ndarray_dim_as_int,
                        is_slice, is_ellipsis, is_ndarray_of_bools,
                        is_list_of_int, is_ndsequence_of_ints,
                        is_ndsequence_of_bools, ndsequence_dim_as_int,
                        is_basic_index_sequence, is_none, dim_as_type,
                        DIMTYPE_TO_INT)


def ndarray_getitem(bound_args: Dict[str, BoundArgument],
                    ctx: FunctionContext):

    self_type = ctx.type
    assert len(bound_args) == 1
    index_arg = next(iter(bound_args.values()))
    index_arg_typ = index_arg.arg_typ
    self_ndim_name = self_type.args[1].type.name()

    if self_ndim_name not in DIMTYPE_TO_INT:
        return ctx.default_return_type

    self_ndim_int = DIMTYPE_TO_INT[self_ndim_name]

    try:
        result_ndim = basic_indexing_ndim(self_ndim_int, index_arg_typ)
    except BasicIndexingError:
        result_ndim = advanced_indexing_ndim(self_ndim_int, index_arg_typ)

    if result_ndim == 0:
        return ctx.default_return_type.args[0]
    return ctx.default_return_type.copy_modified(
        args=[ctx.default_return_type.args[0], dim_as_type(result_ndim)])


def basic_indexing_ndim(input_ndim: int, type: Type) -> Union[int, str]:
    if is_int(type):
        return input_ndim - 1
    if is_slice(type):
        return input_ndim
    if is_ellipsis(type):
        return input_ndim
    if is_none(type):
        return input_ndim + 1
    if isinstance(type, TupleType) and is_basic_index_sequence(type):
        s = input_ndim
        for i in type.items:
            this_dim = basic_indexing_ndim(0, i)
            if isinstance(this_dim, int):
                s += this_dim
            else:
                s = this_dim
        return s

    raise BasicIndexingError()


def advanced_indexing_ndim(input_ndim, type):
    if is_ndarray_of_ints(type, no_bools=True):
        return input_ndim + ndarray_dim_as_int(type) - 1

    if is_ndarray_of_bools(type):
        n_int_arrays = ndarray_dim_as_int(type)  # like .nonzero()
        n_slices = input_ndim - n_int_arrays
        return 1 + n_slices

    if is_list_of_int(type):
        return input_ndim

    if isinstance(type, TupleType) and all(
            is_int(i) or is_slice(i) or is_ellipsis(i) or is_ndarray_of_ints(i)
            or is_ndsequence_of_ints(i) or is_ndarray_of_bools(i)
            or is_ndsequence_of_bools(i) for i in type.items):

        n_ellipsis = sum(is_ellipsis(i) for i in type.items)
        n_int_arrays = sum(is_ndarray_of_ints(i, no_bools=True) or is_ndsequence_of_ints(i, no_bools=True) for i in type.items)
        n_bool_arrays = sum(is_ndarray_of_bools(i) or is_ndsequence_of_bools(i) for i in type.items)
        n_slices = sum(is_slice(i) for i in type.items)
        n_ints = sum(is_int(i) for i in type.items)

        n_effective_int_arrays = (n_int_arrays +
            max((ndarray_dim_as_int(i) for i in type.items if is_ndarray_of_bools(i)), default=0) +
            max((ndsequence_dim_as_int(i) for i in type.items if is_ndsequence_of_bools(i)), default=0))
        int_array_broadcast_dim = max(chain(
                (ndarray_dim_as_int(i) for i in type.items if is_ndarray_of_ints(i)),
                (ndsequence_dim_as_int(i) for i in type.items if is_ndsequence_of_ints(i))),
                default=1)

        if n_effective_int_arrays + n_slices < (input_ndim - n_ints):
            n_slices += (input_ndim - n_effective_int_arrays + n_slices - n_ints)
        assert n_slices + n_effective_int_arrays == input_ndim - n_ints

        return int_array_broadcast_dim + n_slices

    return 'Any'


class BasicIndexingError(Exception):
    pass
