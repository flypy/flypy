# -*- coding: utf-8 -*-

"""
Simplify untyped IR.
"""

from __future__ import print_function, division, absolute_import
import operator

from numba2.compiler.special import lookup_special

from pykit import types
from pykit.ir import Const, Op

#===------------------------------------------------------------------===
# Simplifiers
#===------------------------------------------------------------------===

special_runtime = {
    operator.floordiv: '__floordiv__',
    operator.truediv:  '__truediv__',
}

def rewrite_ops(func, env=None):
    """
    Rewrite unary/binary operations to special methods:

        pycall(operator.add, a, b) -> call(getfield(a, __add__), [a, b])
    """
    for op in func.ops:
        if op.opcode == 'call':
            f, args = op.args
            if not isinstance(f, Const):
                continue
            else:
                f = f.const

            try:
                methname = lookup_special(f)
            except KeyError:
                if f in special_runtime:
                    methname = special_runtime[f]
                else:
                    continue

            self = args[0]
            args = args[1:]
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