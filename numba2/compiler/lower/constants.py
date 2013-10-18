# -*- coding: utf-8 -*-

"""
Handle constants.
"""

from __future__ import print_function, division, absolute_import

from numba2.runtime import fromobject, toctypes

from pykit.utils.ctypes import from_ctypes_value
from pykit.ir import collect_constants, substitute_args

#===------------------------------------------------------------------===
# Constant rewriting
#===------------------------------------------------------------------===

_keep_alive = []

def rewrite_constants(func, env):
    """
    Rewrite constants with user-defined types to IR constants. Also rewrite
    constants of builtins to instances of numba classes.

        e.g. constant(None)  -> constant(NoneValue)
             constant("foo") -> constant(Bytes("foo"))
    """
    context = env['numba.typing.context']

    for op in func.ops:
        constants = collect_constants(op)
        new_constants = []
        for c in constants:
            ty = context[c]

            # Python -> Numba (if not already)
            numba_obj = fromobject(c.const, ty)
            # Numba -> ctypes
            ctype_obj = toctypes(numba_obj, ty, _keep_alive)
            # ctypes -> pykit
            new_const = from_ctypes_value(ctype_obj)

            context[new_const] = ty
            new_constants.append(new_const)

        substitute_args(op, constants, new_constants)