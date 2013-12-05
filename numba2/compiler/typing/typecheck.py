# -*- coding: utf-8 -*-

"""
Type checking after type inference.
"""

from __future__ import print_function, division, absolute_import

from numba2.types import Function, ForeignFunction

from pykit.ir import visit, Undef
from pykit.utils import flatten

#===------------------------------------------------------------------===
# Type checking
#===------------------------------------------------------------------===

class TypeChecker(object):

    def __init__(self, context):
        self.context = context

    def op_setfield(self, op):
        obj, attr, value = op.args
        obj_type = self.context[obj]
        if attr not in obj_type.fields and attr not in obj_type.layout:
            raise TypeError(
                "Object of type '%s' has no attribute %r" % (obj_type, attr))

#===------------------------------------------------------------------===
# Scoping
#===------------------------------------------------------------------===

def check_scoping(func, env):
    for op in func.ops:
        if op.opcode != 'phi':
            for arg in flatten(op.args):
                if isinstance(arg, Undef):

                    raise NameError("Variable referenced before assignment")

#===------------------------------------------------------------------===
# Entry point
#===------------------------------------------------------------------===

def typecheck(func, env):
    context = env['numba.typing.context']
    visit(TypeChecker(context), func)

run = typecheck