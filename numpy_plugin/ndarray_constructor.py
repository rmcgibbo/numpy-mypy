from typing import Tuple, Dict
from mypy.plugin import FunctionContext
from .bind_arguments import BoundArgument
from .shortcuts import (is_int, is_ndarray_of_ints, ndarray_dim_as_int,
                        is_slice, is_ellipsis, is_ndarray_of_bools,
                        is_list_of_int, is_ndsequence_of_ints,
                        is_ndsequence_of_bools, ndsequence_dim_as_int,
                        is_basic_index_sequence, is_none,
                        is_ndarray, ndsequence_dim_as_type)


def ndarray_constructor(bound_args: Dict[str, BoundArgument],
                        ctx: FunctionContext):

    assert 'object' in bound_args
    arg_typ = bound_args['object'].arg_typ

    if is_ndarray(arg_typ):
        return arg_typ
    if is_ndsequence_of_ints(arg_typ):
        return ctx.default_return_type.copy_modified(args=
            [ctx.api.named_type('builtins.int'), ndsequence_dim_as_type(arg_typ)])
    if is_ndsequence_of_bool(arg_typ):
        return ctx.default_return_type.copy_modified(args=
            [ctx.api.named_type('builtins.bool'), ndsequence_dim_as_type(arg_typ)])
    if is_ndsequence_of_float(arg_typ):
        return ctx.default_return_type.copy_modified(args=
            [ctx.api.named_type('builtins.float'), ndsequence_dim_as_type(arg_typ)])

    raise ValueError()




