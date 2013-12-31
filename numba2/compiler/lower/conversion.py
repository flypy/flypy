# -*- coding: utf-8 -*-

"""
Lower conversion operations.
"""

from __future__ import print_function, division, absolute_import

import numba2
from numba2.pipeline import environment
from ..utils import Caller

from pykit import types
from pykit.ir import OpBuilder, Const, OConst

def run(func, env):
    """
    Turn `convert` ops into calls to coerce().
    """
    from numba2.runtime.coercion import coerce

    phase = env['numba.state.phase']

    if env['numba.state.opaque']:
        return # TODO: @no_opaque decorator...

    context = env["numba.typing.context"]
    envs = env["numba.state.envs"]

    builder = OpBuilder()
    caller = Caller(builder, context, env)

    for op in func.ops:
        if op.opcode == 'coerce':
            [arg] = op.args

            dst_type = context[op]
            src_type = context[arg]

            if src_type == dst_type:
                continue

            type_argtype = numba2.Type[dst_type]
            type_arg = OConst(dst_type)
            context[type_arg] = type_argtype

            call = caller.call(phase, coerce, [arg, type_arg],
                               result=op.result)
            op.replace(call)