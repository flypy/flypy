# -*- coding: utf-8 -*-

"""
Lower conversion operations.
"""

from __future__ import print_function, division, absolute_import

import flypy
from flypy import errors
from flypy.pipeline import environment
from ..utils import Caller

from pykit import types
from pykit.ir import OpBuilder, Const, OConst

def run(func, env):
    """
    Turn `convert` ops into calls to coerce().
    """
    with errors.errctx(env):
        lower_coerce(func, env)

def lower_coerce(func, env):
    from flypy.runtime.coercion import coerce

    phase = env['flypy.state.phase']

    if env['flypy.state.opaque']:
        return # TODO: @no_opaque decorator...

    context = env["flypy.typing.context"]
    envs = env["flypy.state.envs"]

    builder = OpBuilder()
    caller = Caller(builder, context, env)

    for op in func.ops:
        if op.opcode == 'coerce':
            [arg] = op.args

            dst_type = context[op]
            src_type = context[arg]

            if src_type == dst_type:
                continue

            type_argtype = flypy.Type[dst_type]
            type_arg = OConst(dst_type)
            context[type_arg] = type_argtype

            call = caller.call(phase, coerce, [arg, type_arg],
                               result=op.result)
            op.replace(call)