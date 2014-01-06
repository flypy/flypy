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
    # NOTE: special cases should be kept to an absolute minimum here. They
    #       should probably be introduced only if ctypes cannot represent the
    #       type

    from numba2.lib import vectorobject

    if ty.impl == vectorobject.Vector:
        # Ctypes does not support vectors
        base, count = ty.parameters
        return ptypes.Vector(representation_type(base), count)

    cty = conversion.ctype(ty)
    result_type = ctypes_support.from_ctypes_type(cty)
    if result_type.is_struct:
        result_type = ptypes.Pointer(result_type)

    return result_type
