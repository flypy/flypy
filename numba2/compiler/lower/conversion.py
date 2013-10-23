# -*- coding: utf-8 -*-

"""
Convert the return value to the function return type.
"""

from __future__ import print_function, division, absolute_import
from pykit.ir import Builder

def convert_retval(func, env):
    """
    Rewrite 'return x' to 'return (restype) x'
    """
    if env['numba.state.opaque']:
        return

    restype = func.type.restype
    context = env['numba.typing.context']

    b = Builder(func)
    for op in func.ops:
        if op.opcode != 'ret' or op.args[0] is None:
            continue

        [retval] = op.args
        if retval.type != restype:
            b.position_before(op)
            converted = b.convert(restype, retval)
            op.set_args([converted])

            # Update type context
            context[converted] = context[retval]