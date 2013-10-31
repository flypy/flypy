# -*- coding: utf-8 -*-

"""
Handle calling conventions for objections.
"""

from __future__ import print_function, division, absolute_import

from numba2.types import Pointer
from numba2.runtime import conversion

from pykit import types
from pykit.ir import OpBuilder, Builder

#===------------------------------------------------------------------===
# Return Objects
#===------------------------------------------------------------------===

opaque_t = types.Pointer(types.Opaque)

def rewrite_obj_return(func, env):
    """
    Handle returning stack-allocated objects.
    """
    if env['numba.state.opaque']:
        return

    context = env['numba.typing.context']
    restype = env['numba.typing.restype']

    if not conversion.stack_allocate(restype):
        return

    builder = Builder(func)
    out = func.add_arg(func.temp("out"), opaque_t)
    context[out] = Pointer[restype]

    func.type = types.Function(types.Void, func.type.argtypes)

    for op in func.ops:
        if op.opcode == 'ret' and op.args[0] is not None:
            # ret val -> store (load val) out ; ret void
            [val] = op.args
            builder.position_before(op)
            newval = builder.load(val)
            builder.store(newval, out, result=op.result)
            op.set_args([None])

            # Update context
            context[newval] = context[val]

        elif op.opcode == 'call' and op.type != types.Void:
            ty = context[op]
            if conversion.stack_allocate(ty):
                builder.position_before(op)
                retval = builder.alloca(opaque_t)
                builder.position_after(op)
                op.replace_uses(retval)
                #op.replace_uses(builder.load(retval_p))

                # Update context
                context[retval] = Pointer[context[op]]