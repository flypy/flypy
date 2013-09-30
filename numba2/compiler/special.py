# -*- coding: utf-8 -*-

"""
Simple definitions to work with special methods.
"""

from __future__ import print_function, division, absolute_import

from pykit.ir import ops, defs
from pykit.utils import invert

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
    ops.le            : '__le__',
    ops.gt            : '__gt__',
    ops.ge            : '__ge__',
    ops.eq            : '__eq__',
    ops.ne            : '__ne__',
    ops.contains      : '__contains__',
}
special2op = invert(special)

def lookup_special(func):
    """Look up a special method name for an operator.* function"""
    operator = defs.operator2opcode[func]
    return special[operator]

def lookup_operator(name):
    """Given a special __*__ name, return the operator.* function"""
    opcode = special2op[name]
    return defs.opcode2operator[opcode]
