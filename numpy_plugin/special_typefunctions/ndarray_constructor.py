from typing import Dict
from mypy.types import AnyType
from mypy.plugin import FunctionContext
import logging
from ..bind_arguments import BoundArgument
from ..shortcuts import (is_ndarray, is_ndsequence_of_ints, is_ndsequence_of_bools, is_ndsequence_of_floats,
                         ndsequence_dim_as_type)

log = logging.getLogger(__name__)


def ndarray_constructor(bound_args: Dict[str, BoundArgument],
                        ctx: FunctionContext):

    assert 'object' in bound_args
    arg_typ = bound_args['object'].arg_typ

    if is_ndarray(arg_typ):
        print('4', ctx.context.line)
        return arg_typ
    if is_ndsequence_of_ints(arg_typ):
        print('1', ctx.context.line)
        return ctx.default_return_type.copy_modified(args=
            [ctx.api.named_type('builtins.int'), ndsequence_dim_as_type(arg_typ)])
    if is_ndsequence_of_bools(arg_typ):
        print('2', ctx.context.line, arg_typ)
        return ctx.default_return_type.copy_modified(args=
            [ctx.api.named_type('builtins.bool'), ndsequence_dim_as_type(arg_typ)])
    if is_ndsequence_of_floats(arg_typ):
        print('3')
        return ctx.default_return_type.copy_modified(args=
            [ctx.api.named_type('builtins.float'), ndsequence_dim_as_type(arg_typ)])

    ctx.api.fail('Could not determine type', ctx.context)
    return ctx.default_return_type.copy_modified(args=[AnyType(), AnyType()])
