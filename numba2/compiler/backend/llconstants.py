# -*- coding: utf-8 -*-

"""
Handle low-level constants after we have resolved the low-level types.
"""

from __future__ import print_function, division, absolute_import

from pykit.ir import Const, Struct, Builder, collect_constants, substitute_args

#===------------------------------------------------------------------===
# Pass
#===------------------------------------------------------------------===

def rewrite_lowlevel_constants(func, env):
    """
    Rewrite constant pointers.
    """
    b = Builder(func)
    b.position_at_beginning(func.startblock)

    for op in func.ops:
        constants = collect_constants(op)
        new_constants = []
        for c in constants:
            new_constants.append(allocate_pointer_const(b, c))
        substitute_args(op, constants, new_constants)


def allocate_pointer_const(b, const):
    ty = const.type
    val = const.const
    if ty.is_pointer:
        value = allocate_pointer_const(b, Const(val.base, ty.base))
        variable = b.alloca(ty)
        b.store(value, variable)
        return variable
    return const
