# -*- coding: utf-8 -*-

"""
Handle constants.
"""

from __future__ import print_function, division, absolute_import

from numba2.types import Int, Float

from pykit.ir import Const, Struct
from pykit.utils import nestedmap

#===------------------------------------------------------------------===
# Helpers
#===------------------------------------------------------------------===

def _collect_constants(x):
    if isinstance(x, Const):
        return x

def build_struct_value(value, seen=None):
    """
    Build a constant struct value from the given runtime Python
    user-defined object.
    """
    seen = seen or set()
    if id(value) in seen:
        raise TypeError("Cannot use recursive value as a numba constant")
    seen.add(id(value))

    cls = type(value)
    names, types = zip(*cls.fields)
    values = [getattr(value, name) for name in names]
    return Struct(names, values)

#===------------------------------------------------------------------===
# Passes
#===------------------------------------------------------------------===

def rewrite_constants(func, env):
    """
    Rewrite constants with user-defined types to IR constants.
    """
    context = env['numba.typing.context']

    for op in func.ops:
        constants = nestedmap(_collect_constants, op.args)
        new_constants = []
        for c in constants:
            ty = context[c]
            value = c
            if type(ty) not in (Int, Float):
                assert isinstance(c.const, ty)
                value = build_struct_value(value)
                context[value] = ty

            new_constants.append(value)

        if constants != new_constants:
            replacements = dict(zip(constants, new_constants))
            new_args = nestedmap(lambda x: replacements.get(x, x), op.args)
            op.set_args(new_args)