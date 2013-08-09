# -*- coding: utf-8 -*-

"""
Support for special methods. See also simplify.py.
"""

from __future__ import print_function, division, absolute_import

from .. import InferError

from pykit.ir import ops

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

def lookup_special(op, type):
    """Look up special method on a type for a given opcode"""
    name = special[op]
    if name in type.methods:
        return type.methods[name]
    else:
        raise InferError("%d: Type %s has no method %s" % (op.lineno, type, name))
