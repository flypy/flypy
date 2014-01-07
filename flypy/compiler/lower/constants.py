# -*- coding: utf-8 -*-

"""
Handle constants.
"""

from __future__ import print_function, division, absolute_import
import ctypes

from flypy.representation import byref
from flypy.conversion import fromobject, toctypes

from pykit.utils.ctypes_support import from_ctypes_value
from pykit.ir import collect_constants, substitute_args

#===------------------------------------------------------------------===
# Constant rewriting
#===------------------------------------------------------------------===

_keep_alive = []

def rewrite_constants(func, env):
    """
    Rewrite constants with user-defined types to IR constants. Also rewrite
    constants of builtins to instances of flypy classes.

        e.g. constant(None)  -> constant(NoneValue)
             constant("foo") -> constant(Bytes("foo"))
    """
    if env['flypy.state.opaque']:
        return

    context = env['flypy.typing.context']

    for op in func.ops:
        if op.opcode == 'exc_catch':
            continue
        constants = collect_constants(op)
        new_constants = []
        for c in constants:
            ty = context[c]

            # Python -> flypy (if not already)
            flypy_obj = fromobject(c.const, ty)
            # flypy -> ctypes
            ctype_obj = toctypes(flypy_obj, ty, _keep_alive)
            if byref(ty):
                ctype_obj = ctypes.pointer(ctype_obj)
            # ctypes -> pykit
            new_const = from_ctypes_value(ctype_obj)

            context[new_const] = ty
            new_constants.append(new_const)

            _keep_alive.extend([ctype_obj, c.const])

        substitute_args(op, constants, new_constants)