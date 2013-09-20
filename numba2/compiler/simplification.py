# -*- coding: utf-8 -*-

"""
Simplify untyped IR.
"""

from __future__ import print_function, division, absolute_import

from .special import lookup_special

from pykit import types
from pykit.ir import Builder, OpBuilder, Const, Op, ops, defs

#===------------------------------------------------------------------===
# Special methods
#===------------------------------------------------------------------===

special = {
    # Unary
    ops.invert        : '__invert__',
    ops.uadd          : '__pos__',
    ops.usub          : '__neg__',

    # Binary
    ops.add           : '__add__',
    ops.sub           : '__sub__',
    ops.mul           : '__mul__',
    ops.div           : '__div__',
    ops.mod           : '__mod__',
    ops.lshift        : '__lshift__',
    ops.rshift        : '__rshift__',
    ops.bitor         : '__or__',
    ops.bitand        : '__and__',
    ops.bitxor        : '__xor__',

    # Compare
    ops.lt            : '__lt__',
    ops.lte           : '__le__',
    ops.gt            : '__gt__',
    ops.gte           : '__ge__',
    ops.eq            : '__eq__',
    ops.noteq         : '__ne__',
    ops.contains      : '__contains__',
    }

def lookup_special(func):
    """Look up a special method name for an operator.* function"""
    operator = defs.func2operator[func]
    return special[operator]

#===------------------------------------------------------------------===
# Simplifier
#===------------------------------------------------------------------===

def rewrite_ops(func, env):
    """
    Rewrite unary/binary operations to special methods:

        pycall(operator.add, a, b) -> call(getfield(a, __add__), [a, b])
    """
    for op in func.ops:
        if op.opcode == 'pycall':
            f, args = op.args[0], op.args[1:]
            if not isinstance(f, Const):
                continue
            else:
                f = f.const

            try:
                methname = lookup_special(f)
            except KeyError:
                continue

            self = args[0]
            m = newop('getfield', [self, methname])
            call = newop('call', [m, args])
            op.replace_uses(call)
            op.replace([m, call])

run = rewrite_ops

#===------------------------------------------------------------------===
# Helpers
#===------------------------------------------------------------------===

def newop(opcode, args):
    return Op(opcode, types.Opaque, args)