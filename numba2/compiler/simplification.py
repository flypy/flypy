# -*- coding: utf-8 -*-

"""
Simplify untyped IR.
"""

from __future__ import print_function, division, absolute_import
import operator

from numba2.typing import overlay_registry
from numba2.compiler.special import lookup_special

from pykit import types
from pykit.ir import Const, Op, collect_constants, substitute_args
from pykit.utils import hashable

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
        if op.opcode == 'call' and isinstance(op.args[0], Const):
            f, args = op.args
            f = f.const
            if not hashable(f):
                continue

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

def rewrite_overlays(func, env=None):
    """
    Resolve overlays of constants.
    """
    for op in func.ops:
        consts = collect_constants(op)
        new = []
        for c in consts:
            overlain = overlay_registry.lookup_overlay(c.const)
            if overlain:
                c = Const(overlain, type=c.type)
            new.append(c)
        substitute_args(op, consts, new)

#===------------------------------------------------------------------===
# Helpers
#===------------------------------------------------------------------===

def newop(opcode, args):
    return Op(opcode, types.Opaque, args)