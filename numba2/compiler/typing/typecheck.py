# -*- coding: utf-8 -*-

"""
Type checking after type inference.
"""

from __future__ import print_function, division, absolute_import

from numba2.types import Function, ForeignFunction
from pykit.ir import visit

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

    def op_call(self, op):
        f, args = op.args
        ty = self.context[f]

        assert ty.impl in (Function, ForeignFunction)
        argtypes = ty.parameters[:-1]
        #restype  = ty.parameters[-1]

        if len(args) != len(argtypes):
            raise TypeError("Function %s requires %d argument(s), got %d" % (
                                                f, len(argtypes), len(args)))
        for argtype, arg in zip(argtypes, args):
            got_argtype = self.context[arg]
            if argtype != got_argtype:
                raise TypeError("Expected argument of type %s, got %s" % (
                                                        argtype, got_argtype))

#===------------------------------------------------------------------===
# Entry point
#===------------------------------------------------------------------===

def typecheck(func, env):
    context = env['numba.typing.context']
    visit(TypeChecker(context), func)

run = typecheck