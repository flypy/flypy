# -*- coding: utf-8 -*-

"""
Object layout.
"""

from __future__ import print_function, division, absolute_import

from numba2 import conversion

from pykit import types as ptypes
from pykit.utils import ctypes_support

#===------------------------------------------------------------------===
# Types
#===------------------------------------------------------------------===

def representation_type(ty):
    """
    Get the low-level representation type for a high-level (user-defined) type.

    Returns
    =======
    The pykit type for the object layout.
    """
    from numba2.lib import vectorobject
    from numba2.lib import arrayobject
    from numba2.runtime.obj import pointerobject

    if ty.impl == pointerobject.Pointer:
        (base,) = ty.parameters
        return ptypes.Pointer(representation_type(base))
    if ty.impl == vectorobject.Vector:
        base, count = ty.parameters
        return ptypes.Vector(representation_type(base), count)
    if ty.impl == arrayobject.Array:
        base, count = ty.parameters
        return ptypes.Array(representation_type(base), count)

    cty = conversion.ctype(ty)
    result_type = ctypes_support.from_ctypes_type(cty)
    if result_type.is_struct:
        result_type = ptypes.Pointer(result_type)

    return result_type
