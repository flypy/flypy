# -*- coding: utf-8 -*-

"""
Simplify untyped IR.
"""

from __future__ import print_function, division, absolute_import

from .special import special

from pykit.types import Opaque
from pykit.ir import Op, Const

def newop(opcode, args):
    return Op(opcode, Opaque, args)

def simplify(func):
    """
    Starting point before typing. Simplify the IR by calling special methods.
    """
    for op in func.ops:
        if op.opcode in special:
            methname = special[op.opcode]
            value = op.args[0]
            m = newop('getfield', [value, methname])
            call = newop('call', [m, op.args])
            op.replace_uses(call)
            op.replace([m, call])