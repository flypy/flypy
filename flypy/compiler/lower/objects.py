# -*- coding: utf-8 -*-

"""
Handle calling conventions for objects.
"""

from __future__ import print_function, division, absolute_import

import ctypes

from flypy import jit, void, representation, conversion
from flypy.types import Pointer
from flypy.compiler import is_flypy_cc

from pykit import types
from pykit.ir import OpBuilder, Builder

#===------------------------------------------------------------------===
# Return Objects
#===------------------------------------------------------------------===

opaque_t = types.Pointer(types.Opaque)

@jit('StackVar[a]')
class StackVar(object):
    """
    Represent the loaded stack layout of a value.
    """

    layout = []

    @classmethod
    def ctype(cls, ty):
        cty = conversion.ctype(ty.parameters[0])
        # Get the base type if a pointer
        if hasattr(cty, '_type_'):
            return cty._type_
        return cty

def should_skip(env):
    return env['flypy.state.opaque']

def rewrite_obj_return(func, env):
    """
    Handle returning stack-allocated objects.
    """
    if should_skip(env):
        return

    context = env['flypy.typing.context']
    restype = env['flypy.typing.restype']
    envs =  env['flypy.state.envs']

    builder = Builder(func)

    stack_alloc = representation.byref(restype)

    if stack_alloc:
        out = func.add_arg(func.temp("out"), opaque_t)
        context[out] = Pointer[restype]
        func.type = types.Function(types.Void, func.type.argtypes, False)

    for arg in func.args:
        arg.type = opaque_t
    func.type = types.Function(func.type.restype, (opaque_t,) * len(func.args),
                               False)

    is_generator = env['flypy.state.generator']
    for op in func.ops:
        if (op.opcode == 'ret' and op.args[0] is not None and
                stack_alloc and not is_generator):
            # ret val =>
            #     store (load val) out ; ret void
            [val] = op.args
            builder.position_before(op)
            newval = builder.load(val)
            builder.store(newval, out)
            op.set_args([None])

            # Update context
            context[newval] = StackVar[context[val]]

        elif op.opcode == 'call' and op.type != types.Void:
            # result = call(f, ...) =>
            #     alloca result ; call(f, ..., &result)
            ty = context[op]
            if conversion.byref(ty):
                f, args = op.args
                if not is_flypy_cc(f) or should_skip(envs[f]):
                    continue

                builder.position_before(op)
                retval = builder.alloca(opaque_t)
                builder.position_after(op)
                op.replace_uses(retval)

                newargs = args + [retval]
                op.set_args([f, newargs])

                # Update context
                context[retval] = context[op]
                context[op] = void
