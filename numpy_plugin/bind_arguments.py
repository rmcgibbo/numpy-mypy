from typing import Dict
from collections import namedtuple

BoundArgument = namedtuple('BoundArgument', ('name', 'formal_typ', 'arg_typ', 'arg',))


def bind_arguments(callee, ctx, calltype: str='function') -> Dict[str, BoundArgument]:
    name2arg = {}
    if calltype == 'method':
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
