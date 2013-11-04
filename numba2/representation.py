# -*- coding: utf-8 -*-

"""
Value representation of instances of user-defined types.
"""

from __future__ import print_function, division, absolute_import
from numba2.runtime.conversion import ctype

from pykit import types as ptypes
from pykit.utils.ctypes_support import from_ctypes_type

#===------------------------------------------------------------------===
# Type Representation
#===------------------------------------------------------------------===

def representation_type(ty):
    """
    Get the low-level representation type for a high-level (user-defined) type.
    """
    cty = ctype(ty)
    result_type = from_ctypes_type(cty)
    if result_type.is_struct:
        result_type = ptypes.Pointer(result_type)
    return result_type

lltype = representation_type